"""Detect risk signals in each company's timeline with an OpenAI model.

Reads  prompt/Prompt_<version>.txt
       data/timeline/company_<company_id>.json
Writes data/runs/<run name>/company_<company_id>.json - every run in its own
       folder, so runs of different models and prompts can be compared.

Usage:
    python detect_signals.py                                # all companies, defaults
    python detect_signals.py 176676003                      # one company
    python detect_signals.py --model gpt-5-nano --prompt v1 --effort low
    python detect_signals.py --repeat 3                     # 3 runs, to check consistency
    python detect_signals.py --years 3                      # look back 3 years instead of 2
    python detect_signals.py --since 2025-01-01             # or from a fixed date

Only recent activity is sent to the model (default: the last 2 years). The
timelines themselves keep the full history.
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent
PROMPT_DIR = ROOT / "prompt"
TIMELINE_DIR = ROOT / "data" / "timeline"
RUNS_DIR = ROOT / "data" / "runs"

# Defaults - each can be overridden on the command line (--prompt, --effort,
# --model). Every prompt version is kept in prompt/.
DEFAULT_PROMPT_VERSION = "v2"
# GPT-5 models think before answering, and that thinking is billed as output.
# "medium" found signals that "low" missed.
DEFAULT_EFFORT = "medium"
# Signals are judged on recent activity only: the last N years before today.
DEFAULT_YEARS = 2
# Hard cap on thinking + answer, so one bad call can't run up a big bill.
MAX_OUTPUT_TOKENS = 16_000

# Flex tier: half the price of standard, but slower, and OpenAI may answer
# "429 resource unavailable" when busy (not charged). So we wait longer for
# each answer and let the client retry a few times, backing off in between.
SERVICE_TIER = "flex"
TIMEOUT_SECONDS = 15 * 60
MAX_RETRIES = 5

# --------------------------------------------------------------------------- #
# Reading the prompt: it defines the signals, the input fields and the answer #
# shape, so a new prompt version needs no code change.                        #
# --------------------------------------------------------------------------- #

HEADING = re.compile(r"^[A-Z][A-Z ]{2,}$", re.M)            # e.g. "SIGNALS", "OUTPUT"
SIGNAL_LINE = re.compile(r"^\s*\d+\.\s+([a-z][a-z0-9_]*)\s*$", re.M)   # "1. churn_language"
FIELD_LINE = re.compile(r"^-\s+([a-z][a-z0-9_]*)\s*:", re.M)             # "- ref   : ..."
JSON_KEY = re.compile(r'"([a-z][a-z0-9_]*)"\s*:')


def sections(prompt):
    """{"SIGNALS": "...text...", "OUTPUT": "...", ...} split at the ALL-CAPS headings."""
    marks = list(HEADING.finditer(prompt))
    return {m.group(0).strip(): prompt[m.end():nxt.start() if nxt else len(prompt)]
            for m, nxt in zip(marks, marks[1:] + [None])}


def read_prompt(prompt):
    """What the prompt asks for: signal names, input fields, evidence fields."""
    parts = sections(prompt)
    signals = SIGNAL_LINE.findall(parts.get("SIGNALS", ""))

    # Input fields: the "- name : meaning" list that follows "Each item has:".
    after = prompt.split("Each item has:", 1)[1] if "Each item has:" in prompt else ""
    fields = FIELD_LINE.findall(after.split("\n\n", 1)[0])

    # Evidence fields: the keys inside the first "evidence": [ { ... } ] of OUTPUT.
    output = parts.get("OUTPUT", "")
    first_evidence = output.split('"evidence"', 1)[1].split("]", 1)[0] if '"evidence"' in output else ""
    evidence = JSON_KEY.findall(first_evidence)

    missing = [n for n, v in (("signals (SIGNALS: '1. name')", signals),
                              ("input fields ('Each item has:' list)", fields),
                              ("evidence fields (OUTPUT example)", evidence)) if not v]
    if missing:
        raise ValueError(f"could not read from the prompt: {', '.join(missing)}")
    return {"signals": signals, "fields": fields, "evidence": evidence}


def answer_schema(spec):
    """The JSON the prompt asks for, as a schema the API enforces (strict mode).
    The model cannot return anything that doesn't fit it."""
    evidence = {
        "type": "object",
        "properties": {f: {"type": "integer" if f == "id" else "string"} for f in spec["evidence"]},
        "required": spec["evidence"],
        "additionalProperties": False,
    }
    signal = {
        "type": "object",
        "properties": {"evidence": {"type": "array", "items": evidence},
                       "present": {"type": "boolean"}},
        "required": ["evidence", "present"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {name: signal for name in spec["signals"]},
        "required": spec["signals"],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Preparing the input                                                          #
# --------------------------------------------------------------------------- #

def prompt_file(version):
    return PROMPT_DIR / f"Prompt_{version}.txt"


def load_timeline(company_id):
    path = TIMELINE_DIR / f"company_{company_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_activity(activity, fields):
    """Keep only the fields the prompt describes; drop empty ones so every token
    carries meaning. Anything the prompt doesn't list never reaches the model."""
    out = {}
    for field in fields:
        v = activity.get(field)
        if v in (None, "", False):        # no subject, not truncated, ...
            continue
        out[field] = v
    return out


def prepare_input(timeline, fields, since):
    """The JSON text the model will read: {"activities": [...]}, oldest first.

    Only activities dated on or after `since` (YYYY-MM-DD) with some text.
    Ids are kept as they are in the timeline, so they still match it.
    """
    activities = [prepare_activity(a, fields) for a in timeline["activities"]
                  if a["text"].strip() and (a["date"] or "") >= since]
    # separators=(",", ":") drops the spaces json.dumps adds by default.
    data = json.dumps({"activities": activities}, ensure_ascii=False, separators=(",", ":"))
    return data, len(activities)


# --------------------------------------------------------------------------- #
# Checking the model's answer against the data                                 #
# --------------------------------------------------------------------------- #

def normalise(text):
    """Ignore differences in spacing and line breaks, and punctuation at the ends."""
    return re.sub(r"\s+", " ", text).strip().strip(".,;:!?\"'“”‘’ ")


NOT_FOUND = "activity not in the data"
QUOTE_MISSING = "quote not found in the activity's text"
SERIOUS = (NOT_FOUND, QUOTE_MISSING)          # these make a piece of evidence invalid


def find_activity(evidence, by_ref, by_id):
    """The cited activity: by ref when the answer has one (stable), else by id."""
    if evidence.get("ref") in by_ref:
        return by_ref[evidence["ref"]]
    return by_id.get(evidence.get("id"))


def check_evidence(evidence, by_ref, by_id):
    """List what's wrong with one piece of evidence. An empty list means it's good."""
    activity = find_activity(evidence, by_ref, by_id)
    if activity is None:
        return [NOT_FOUND]

    problems = []
    if "id" in evidence and "ref" in evidence and evidence["id"] != activity["id"]:
        problems.append(f"id should be {activity['id']} for this ref")
    evidence["ref"] = activity["ref"]          # the original long id, for tracing back
    evidence["level"] = activity.get("level")  # "store" or "group"
    evidence["store_ids"] = activity.get("store_ids", [])

    if normalise(evidence.get("quote", "")) not in normalise(activity["text"]) or not evidence.get("quote"):
        problems.append(QUOTE_MISSING)
    if activity["role"] == "adu" and activity["kind"] == "email":
        # Allowed only as a recap of what the customer told us - worth a human look.
        problems.append("our own outbound email - check it restates the customer")
    if "happened_on" in evidence and evidence["happened_on"] != (activity["date"] or "")[:10]:
        problems.append(f"date should be {(activity['date'] or '')[:10]}")
    if "stores" in evidence and evidence["stores"] != activity.get("stores", ""):
        problems.append("stores does not match the activity")
    return problems


def verify(signals, timeline, signal_names):
    """Add a check to every piece of evidence, and a 'confirmed' flag per signal.

    confirmed = at least one piece of evidence points at a real activity whose
    text contains the quote. Other problems (wrong date, our own email) are
    noted for review but don't sink the evidence.
    """
    by_ref = {a["ref"]: a for a in timeline["activities"]}
    by_id = {a["id"]: a for a in timeline["activities"]}
    for name in signal_names:
        signal = signals[name]
        for evidence in signal["evidence"]:
            evidence["problems"] = check_evidence(evidence, by_ref, by_id)
            evidence["verified"] = not any(p in SERIOUS for p in evidence["problems"])
        signal["confirmed"] = any(e["verified"] for e in signal["evidence"])
        if signal["present"] != bool(signal["evidence"]):
            signal["note"] = "model's 'present' did not match its evidence"
    return signals


def label_with_status(signals, timeline, signal_names):
    """Label the answer with the CRM's account status (not something the model sees).

    * Churn-type signals get churn_type: "churn risk" on a current customer,
      "already lost" on an account the CRM already marks as lost.
    * Evidence tied only to inactive stores (NLOP_...) gets store_active: false.
    """
    status = timeline.get("account_status", {}).get("status")
    active = {s["id"]: s["active"] for s in timeline.get("store_list", [])}
    for name in signal_names:
        signal = signals[name]
        if "churn" in name and signal["confirmed"]:
            signal["churn_type"] = "already lost" if status == "lost" else "churn risk"
        for evidence in signal["evidence"]:
            ids = evidence.get("store_ids") or []
            if ids:
                evidence["store_active"] = any(active.get(i, True) for i in ids)
    return signals


# --------------------------------------------------------------------------- #
# Calling the model                                                            #
# --------------------------------------------------------------------------- #

def detect(client, model, prompt, schema, effort, data):
    """Send one company's activities to the model. Returns (signals, usage)."""
    response = client.responses.create(
        model=model,
        instructions=prompt,                 # the rules: your prompt file
        input=data,                          # the evidence: the activities JSON
        reasoning={"effort": effort},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        service_tier=SERVICE_TIER,
        text={"format": {"type": "json_schema", "name": "signals",
                         "schema": schema, "strict": True}},
    )
    if response.status != "completed":
        reason = getattr(response.incomplete_details, "reason", response.status)
        raise RuntimeError(f"model did not finish: {reason}")
    usage = {"service_tier": response.service_tier,
             "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return json.loads(response.output_text), usage


def save(run_dir, timeline, settings, signals, usage, count):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"company_{timeline['company_id']}.json"
    result = {
        "company_id": timeline["company_id"],
        "company_name": timeline["company_name"],
        "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "account_status": timeline.get("account_status", {}).get("status"),
        **settings,                          # model, prompt, effort, run, since
        "activities_sent": count,
        "usage": usage,
        "signals": signals,
    }
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("companies", nargs="*", type=int,
                        help="company ids to run (default: every timeline)")
    parser.add_argument("--model", help="OpenAI model (default: OPENAI_MODEL in .env)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION,
                        help=f"prompt version, e.g. v1 (default: {DEFAULT_PROMPT_VERSION})")
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["minimal", "low", "medium", "high"],
                        help=f"reasoning effort (default: {DEFAULT_EFFORT})")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS,
                        help=f"only send activity from the last N years (default: {DEFAULT_YEARS})")
    parser.add_argument("--since", help="only send activity from this date on, YYYY-MM-DD "
                                        "(overrides --years)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run everything this many times, each in its own folder")
    return parser.parse_args()


def years_ago(years):
    """The date `years` years before today, as YYYY-MM-DD (29 Feb -> 28 Feb)."""
    today = date.today()
    try:
        return today.replace(year=today.year - years).isoformat()
    except ValueError:
        return today.replace(year=today.year - years, day=28).isoformat()


def next_run_name(base):
    """base_r1, base_r2, ... - the first number not used yet, so nothing is overwritten."""
    n = 1
    while (RUNS_DIR / f"{base}_r{n}").exists():
        n += 1
    return f"{base}_r{n}"


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    model = args.model or os.environ["OPENAI_MODEL"]
    prompt = prompt_file(args.prompt).read_text(encoding="utf-8")
    spec = read_prompt(prompt)                 # signals, input fields, evidence fields
    schema = answer_schema(spec)
    if "quote" not in spec["evidence"]:
        raise ValueError('the OUTPUT example must include "quote" - it is how evidence is checked')
    print(f"Prompt {args.prompt}: signals {spec['signals']}\n"
          f"  sends fields {spec['fields']}\n  evidence fields {spec['evidence']}\n")
    client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    since = args.since or years_ago(args.years)
    print(f"Activity window: {since} to today\n")

    company_ids = args.companies or \
                  [int(p.stem.split("_")[1]) for p in sorted(TIMELINE_DIR.glob("company_*.json"))]

    # Warn about fields the prompt describes but the timelines don't have -
    # the model would be told about them but never see them.
    sample = load_timeline(company_ids[0])["activities"]
    known = set().union(*(a.keys() for a in sample))
    unknown = [f for f in spec["fields"] if f not in known]
    if unknown:
        print(f"WARNING: the prompt lists fields the timelines don't have: {unknown}\n"
              f"  available: {sorted(known)}\n")

    for _ in range(args.repeat):
        run = next_run_name(f"{model}_{args.prompt}_{args.effort}")
        run_dir = RUNS_DIR / run
        settings = {"run": run, "model": model, "prompt": prompt_file(args.prompt).name,
                    "effort": args.effort, "since": since, "signals_requested": spec["signals"]}
        print(f"=== run {run} ===\n")

        for company_id in company_ids:
            timeline = load_timeline(company_id)
            data, count = prepare_input(timeline, spec["fields"], since)
            print(f"{timeline['company_name']}: sending {count} of {timeline['count']} activities "
                  f"(since {since}, ~{len(data) // 4:,} tokens) to {model}...")

            signals, usage = detect(client, model, prompt, schema, args.effort, data)
            signals = verify(signals, timeline, spec["signals"])
            signals = label_with_status(signals, timeline, spec["signals"])
            path = save(run_dir, timeline, settings, signals, usage, count)

            for name in spec["signals"]:
                s = signals[name]
                good = sum(e["verified"] for e in s["evidence"])
                print(f"  {name:<20} present={str(s['present']):<5}  confirmed={str(s['confirmed']):<5}  "
                      f"evidence {good}/{len(s['evidence'])} verified")
                for e in s["evidence"]:
                    if e["problems"]:
                        print(f"      {e['id']}: {'; '.join(e['problems'])}")
            churn = [f"{n}: {signals[n]['churn_type']}" for n in spec["signals"] if "churn_type" in signals[n]]
            if churn:
                print(f"  account: {timeline['account_status']['status']}  ->  {', '.join(churn)}")
            print(f"  tier: {usage['service_tier']}  tokens: {usage['input_tokens']:,} in, "
                  f"{usage['output_tokens']:,} out -> {path.relative_to(ROOT)}\n")
