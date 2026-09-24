"""Score detector runs against the team's answer key.

Reads  data/labels.json                      the answer key
       data/runs/<run>/company_<id>.json     one folder per detector run
       data/timeline/company_<id>.json       to read the text of cited activities

Two levels of scoring:
  * signal   - for each company and signal, did the run say yes/no correctly?
  * evidence - did the run cite the activities the answer key names?

Only evidence that passed the checker (verified) counts - a signal is "yes"
when the run's "confirmed" flag is true.

Usage:
    python evaluate.py                    # every run in data/runs
    python evaluate.py gpt-6-luna_v2_medium_r1
    python evaluate.py --details          # also list every miss and false alarm
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
LABELS_FILE = ROOT / "data" / "labels.json"
RUNS_DIR = ROOT / "data" / "runs"
TIMELINE_DIR = ROOT / "data" / "timeline"

SIGNAL_NAMES = ["competitor_mention", "in_house_intent", "churn_language"]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def text_by_ref(company_id):
    """ref -> normalised text of every activity in the company's timeline."""
    timeline = load_json(TIMELINE_DIR / f"company_{company_id}.json")
    return {a["ref"]: normalise(a["text"]) for a in timeline["activities"]}


def score_run(run_dir, labels):
    """Compare one run to the answer key. Returns counts plus a list of misses."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ev_found": 0, "ev_total": 0}
    notes = []
    for company_id, truth in labels["companies"].items():
        path = run_dir / f"company_{company_id}.json"
        if not path.exists():
            notes.append(f"{truth['name']}: no result in this run")
            continue
        result = load_json(path)["signals"]
        texts = text_by_ref(company_id)

        for name in SIGNAL_NAMES:
            expected = truth[name]["present"]
            said = result[name]["confirmed"]
            cited = {e["ref"] for e in result[name]["evidence"] if e.get("verified")}
            cited_texts = [texts.get(r, "") for r in cited]

            # --- signal level ---
            if expected and said:
                counts["tp"] += 1
            elif expected and not said:
                counts["fn"] += 1
                notes.append(f"MISSED       {truth['name']} - {name}")
            elif said and not expected:
                counts["fp"] += 1
                notes.append(f"FALSE ALARM  {truth['name']} - {name}: cited {sorted(cited)}")
            else:
                counts["tn"] += 1

            # --- evidence level: each labelled activity, found or not ---
            # Found = the run cited that activity, or any activity containing the
            # labelled quote (duplicate notes: a call and its follow-up event
            # carry the same words, the event with an extra title line).
            for ev in truth[name]["evidence"]:
                counts["ev_total"] += 1
                quote = normalise(ev["quote"])
                if ev["ref"] in cited or any(quote in t for t in cited_texts):
                    counts["ev_found"] += 1
                else:
                    notes.append(f"  evidence not cited: {truth['name']} - {name}: "
                                 f"{ev['ref']} \"{ev['quote'][:60]}\"")
    return counts, notes


def pct(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "-"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="run folder names (default: all)")
    parser.add_argument("--details", action="store_true", help="list every miss")
    args = parser.parse_args()

    labels = load_json(LABELS_FILE)
    run_dirs = [RUNS_DIR / r for r in args.runs] or sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())

    print(f"{'run':<34}{'correct':>9}{'missed':>8}{'false alarm':>13}"
          f"{'precision':>11}{'recall':>8}{'evidence found':>16}")
    all_notes = {}
    for run_dir in run_dirs:
        c, notes = score_run(run_dir, labels)
        correct = c["tp"] + c["tn"]
        total = correct + c["fp"] + c["fn"]
        score = f"{correct}/{total}"
        evidence = f"{c['ev_found']}/{c['ev_total']}"
        print(f"{run_dir.name:<34}{score:>9}{c['fn']:>8}{c['fp']:>13}"
              f"{pct(c['tp'], c['tp'] + c['fp']):>11}{pct(c['tp'], c['tp'] + c['fn']):>8}"
              f"{evidence:>16}")
        all_notes[run_dir.name] = notes

    if args.details:
        for run, notes in all_notes.items():
            print(f"\n{run}")
            for n in notes or ["  nothing missed"]:
                print(f"  {n}")
