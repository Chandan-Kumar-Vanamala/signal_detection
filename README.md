# Signal Detection

Finds three risk signals in a dealer account's activity history (calls, notes,
events and emails exported from the Solve360 portal):

- **competitor_mention**: the customer mentions another provider for this work
- **in_house_intent**: the customer plans to do the work themselves
- **churn_language**: the customer signals ending, pausing or not renewing

An OpenAI model reads each account's timeline with the rules in `prompt/`, and
answers in JSON with exact quotes as evidence. The code then checks every quote
against the data, and `evaluate.py` scores runs against a team-written answer key.

## Pipeline

```
data/local/        portal exports, one JSON file per company   (not in git)
     │  build_timeline_local.py
     ▼
data/timeline/     one clean, dated timeline per company        (not in git)
     │  detect_signals.py   + prompt/Prompt_<version>.txt
     ▼
data/runs/<run>/   the model's answer per company, per run      (not in git)
     │  evaluate.py         + data/labels.json (answer key)
     ▼
score table: correct / missed / false alarms / precision / recall
```

## Setup

Requires Python 3.12+ (developed on 3.14).

```bash
git clone https://github.com/Chandan-Kumar-Vanamala/signal_detection.git
cd signal_detection

python3 -m venv .venv                # on Ubuntu you may need: sudo apt install python3-venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env                 # then put your own OpenAI API key in .env
```

### The data (shared separately, never committed)

`data/` holds real customer information, so `.gitignore` keeps it out of git.
Get the `data/` folder through the team's secure share and place it in the
project root:

```
data/
  local/        company_<Name>_<ID>_data_with_activities.json   (portal exports)
  timeline/     company_<ID>.json
  runs/         <model>_<prompt>_<effort>_r<N>/company_<ID>.json
  labels.json   the answer key
```

## Usage

```bash
# 1. Build timelines from the portal exports (after adding or replacing a file in data/local/)
.venv/bin/python build_timeline_local.py

# 2. Detect signals - each run is saved in its own folder under data/runs/
.venv/bin/python detect_signals.py                                # all companies, defaults
.venv/bin/python detect_signals.py 176676003                      # one company
.venv/bin/python detect_signals.py --model gpt-5-nano --prompt v1 --effort low
.venv/bin/python detect_signals.py --repeat 3                     # 3 runs, for consistency

# 3. Score runs against the answer key
.venv/bin/python evaluate.py                    # compare every run
.venv/bin/python evaluate.py --details          # plus every miss and false alarm
```

`test_openai.py` sends one tiny request to check that the API key and model work.

Defaults: model from `OPENAI_MODEL` in `.env` (`gpt-6-luna`), prompt `v2`,
reasoning effort `medium`, OpenAI **flex** tier (half price, slower; the client
retries automatically when OpenAI is busy). A full run over 5 companies costs
roughly 1.5 cents.

## Prompts

Every prompt version is kept in `prompt/`. To try a change, copy the latest
file to the next number (e.g. `Prompt_v3.txt`), edit it, and run with
`--prompt v3`. Each result records which model, prompt and effort produced it.
