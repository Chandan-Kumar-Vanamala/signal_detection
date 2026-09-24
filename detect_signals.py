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
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
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
# Hard cap on thinking + answer, so one bad call can't run up a big bill.
MAX_OUTPUT_TOKENS = 16_000

# Flex tier: half the price of standard, but slower, and OpenAI may answer
# "429 resource unavailable" when busy (not charged). So we wait longer for
# each answer and let the client retry a few times, backing off in between.
SERVICE_TIER = "flex"
TIMEOUT_SECONDS = 15 * 60
MAX_RETRIES = 5

# Only the fields the prompt describes. Everything else (author names and
# addresses, source, seen_in...) stays out: smaller input, and no personal
# details the prompt tells the model never to quote anyway.
FIELDS = ["id", "date", "kind", "role", "stores", "subject", "text", "truncated"]


# --------------------------------------------------------------------------- #
# The answer's shape - the same JSON the prompt asks for, as a schema the API #
# enforces. The model cannot return anything that doesn't fit it.            #
# --------------------------------------------------------------------------- #

EVIDENCE = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "happened_on": {"type": "string"},
        "stores": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["id", "happened_on", "stores", "quote"],
    "additionalProperties": False,
}

SIGNAL = {
    "type": "object",
    "properties": {
        "evidence": {"type": "array", "items": EVIDENCE},
        "present": {"type": "boolean"},
    },
    "required": ["evidence", "present"],
    "additionalProperties": False,
}

SIGNAL_NAMES = ["competitor_mention", "in_house_intent", "churn_language"]

SCHEMA = {
    "type": "object",
    "properties": {name: SIGNAL for name in SIGNAL_NAMES},
    "required": SIGNAL_NAMES,
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


def prepare_activity(activity):
    """Keep the prompt's fields; drop empty ones so every token carries meaning."""
    out = {}
    for field in FIELDS:
        v = activity.get(field)
        if v in (None, "", False):        # no subject, not truncated, ...
            continue
        out[field] = v
    return out


def prepare_input(timeline):
    """The JSON text the model will read: the list of activities, oldest first."""
    activities = [prepare_activity(a) for a in timeline["activities"] if a["text"].strip()]
    # separators=(",", ":") drops the spaces json.dumps adds by default.
    return json.dumps(activities, ensure_ascii=False, separators=(",", ":")), len(activities)


# --------------------------------------------------------------------------- #
# Checking the model's answer against the data                                 #
# --------------------------------------------------------------------------- #

def normalise(text):
    """Ignore differences in spacing and line breaks, and punctuation at the ends."""
    return re.sub(r"\s+", " ", text).strip().strip(".,;:!?\"'“”‘’ ")


def check_evidence(evidence, activities):
    """List what's wrong with one piece of evidence. An empty list means it's good."""
    activity = activities.get(evidence["id"])
    if activity is None:
        return ["activity id not in the data"]

    evidence["ref"] = activity["ref"]          # the original long id, for tracing back
    problems = []
    if normalise(evidence["quote"]) not in normalise(activity["text"]):
        problems.append("quote not found in the activity's text")
    if activity["role"] == "adu" and activity["kind"] == "email":
        problems.append("cites our own outbound email, not the customer")
    if evidence["happened_on"] != (activity["date"] or "")[:10]:
        problems.append(f"date should be {(activity['date'] or '')[:10]}")
    if evidence["stores"] != activity.get("stores", ""):
        problems.append("stores does not match the activity")
    return problems


def verify(signals, timeline):
    """Add a check to every piece of evidence, and a 'confirmed' flag per signal.

    confirmed = at least one piece of evidence passed the quote and source
    checks. A wrong date or stores value alone doesn't sink it - the quote
    being real is what matters.
    """
    activities = {a["id"]: a for a in timeline["activities"]}
    serious = ("activity id not in the data", "quote not found in the activity's text",
               "cites our own outbound email, not the customer")
    for name in SIGNAL_NAMES:
        signal = signals[name]
        for evidence in signal["evidence"]:
            evidence["problems"] = check_evidence(evidence, activities)
            evidence["verified"] = not any(p in serious for p in evidence["problems"])
        signal["confirmed"] = any(e["verified"] for e in signal["evidence"])
        if signal["present"] != bool(signal["evidence"]):
            signal["note"] = "model's 'present' did not match its evidence"
    return signals


# --------------------------------------------------------------------------- #
# Calling the model                                                            #
# --------------------------------------------------------------------------- #

def detect(client, model, prompt, effort, data):
    """Send one company's activities to the model. Returns (signals, usage)."""
    response = client.responses.create(
        model=model,
        instructions=prompt,                 # the rules: your prompt file
        input=data,                          # the evidence: the activities JSON
        reasoning={"effort": effort},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        service_tier=SERVICE_TIER,
        text={"format": {"type": "json_schema", "name": "signals",
                         "schema": SCHEMA, "strict": True}},
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
        **settings,                          # model, prompt, effort, run
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
    parser.add_argument("--repeat", type=int, default=1,
                        help="run everything this many times, each in its own folder")
    return parser.parse_args()


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
    client = OpenAI(timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

    company_ids = args.companies or \
                  [int(p.stem.split("_")[1]) for p in sorted(TIMELINE_DIR.glob("company_*.json"))]

    for _ in range(args.repeat):
        run = next_run_name(f"{model}_{args.prompt}_{args.effort}")
        run_dir = RUNS_DIR / run
        settings = {"run": run, "model": model, "prompt": prompt_file(args.prompt).name,
                    "effort": args.effort}
        print(f"=== run {run} ===\n")

        for company_id in company_ids:
            timeline = load_timeline(company_id)
            data, count = prepare_input(timeline)
            print(f"{timeline['company_name']}: sending {count} activities (~{len(data) // 4:,} tokens) "
                  f"to {model}...")

            signals, usage = detect(client, model, prompt, args.effort, data)
            signals = verify(signals, timeline)
            path = save(run_dir, timeline, settings, signals, usage, count)

            for name in SIGNAL_NAMES:
                s = signals[name]
                good = sum(e["verified"] for e in s["evidence"])
                print(f"  {name:<20} present={str(s['present']):<5}  confirmed={str(s['confirmed']):<5}  "
                      f"evidence {good}/{len(s['evidence'])} verified")
                for e in s["evidence"]:
                    if e["problems"]:
                        print(f"      {e['id']}: {'; '.join(e['problems'])}")
            print(f"  tier: {usage['service_tier']}  tokens: {usage['input_tokens']:,} in, "
                  f"{usage['output_tokens']:,} out -> {path.relative_to(ROOT)}\n")
