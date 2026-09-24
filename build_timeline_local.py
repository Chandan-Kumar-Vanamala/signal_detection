"""Build timelines from the Solve360 portal export files in data/local/.

Reads  data/local/company_<name>_<company_id>_data_with_activities.json
Writes data/timeline/company_<company_id>.json      <- what the LLM will read

Every activity gets a short id (1, 2, 3 ... in date order) plus "ref", its
original long id. The input differs from a database export:
  * one "Activities" list per company, emails included (Type = "Email")
  * email text is already plain text, but has NO line breaks
  * email addresses are masked, e.g. "er***@de**ft.com"
  * the portal only exports the full text of the newest 100 emails; older
    ones carry a ~200-character preview (flagged "truncated")
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
LOCAL_DIR = ROOT / "data" / "local"
TIMELINE_DIR = ROOT / "data" / "timeline"

PORTAL_FULL_EMAIL_LIMIT = 100      # MAX_EMAIL_FETCHES in the portal's export code

KINDS = {"Call Log": "call", "Call": "call", "Note": "note", "Event": "event",
         "Email": "email", "File": "file", "SWAT": "swat", "SWAT-Projects": "swat",
         "Projection": "projection"}


# --------------------------------------------------------------------------- #
# Small helpers: passwords, dates, message fingerprints                        #
# --------------------------------------------------------------------------- #

# Passwords turn up in plain text (dealers sending logins). Never pass them on.
PASSWORD = re.compile(r"\b(password|passwd|pwd)\b(\s*[:=-]?\s*)\S+", re.I)


def redact(text):
    return PASSWORD.sub(r"\1\2[REDACTED]", text)


# Quoted email headers write dates in many ways, e.g.
#   "Thursday, March 21, 2024 11:45 AM"   "Tue, Oct 15, 2024 at 7:04 AM"
#   "Wednesday, August 2, 2023 5:47:00 PM"   "Mar 24, 2023"
DATE_FORMATS = ["%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M:%S %p",
                "%b %d, %Y %I:%M:%S %p", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y %I:%M %p",
                "%m/%d/%y %I:%M %p", "%m/%d/%Y", "%m/%d/%y"]
# Trailing time zones we can safely ignore at day-level accuracy: "PST", "(GMT-06:00)".
TIMEZONE = re.compile(r"\s*(\(?(GMT|UTC)[+-]?[\d:]*\)?|\b[ECMP][SD]T)\s*$", re.I)
WEEKDAY = re.compile(r"^(mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\s*", re.I)


def parse_date(text):
    """Turn any date string we have seen into 'YYYY-MM-DD HH:MM', or None."""
    if not text:
        return None
    text = str(text).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", text):                  # already ISO
        return text[:16]
    text = text.replace("\u202f", " ")                         # odd space in some Mac mail
    text = TIMEZONE.sub("", WEEKDAY.sub("", text))
    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)   # "July 22nd" -> "July 22"
    text = re.sub(r"(\d{4}),?\s+(at\s+)?", r"\1 ", text)       # "2024, at 7:04" -> "2024 7:04"
    text = re.sub(r"\s+", " ", text).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None


def fingerprint(author, text):
    """Same author + same words = same message, however it was wrapped or quoted."""
    normal = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(f"{author}|{normal}".encode()).hexdigest()[:12]


def value(v):
    """The export writes missing values as the text 'None' - treat that as empty."""
    v = "" if v is None else str(v).strip()
    return "" if v in ("None", "null") else v


# --------------------------------------------------------------------------- #
# Who is ADU staff?                                                            #
# --------------------------------------------------------------------------- #

MASKED_ADDRESS = re.compile(r"[\w.+'*-]+@[\w*.-]+\.[a-z]{2,}", re.I)
NAME_AND_ADDRESS = re.compile(r"'?([A-Z][\w.'-]*(?: [A-Z][\w.'-]*)+)'?\s*<\s*([^>]+?)\s*>")


def is_adu_address(address):
    address = address.lower()
    return address.endswith("@dealeruplift.com") or address.endswith("@de**ft.com")


def adu_names(companies):
    """Staff names: everyone in 'Assigned To', plus named ADU addresses in To/CC."""
    names = set()
    for company in companies:
        for a in company["Activities"]:
            if value(a["Assigned To"]):
                names.add(value(a["Assigned To"]).lower())
            for field in ("To", "CC"):
                for name, address in NAME_AND_ADDRESS.findall(value(a[field])):
                    if is_adu_address(address):
                        names.add(name.lower())
    return names


def role_of(author, staff):
    if not author:
        return "unknown"
    found = MASKED_ADDRESS.search(author)
    if found:
        return "adu" if is_adu_address(found.group(0)) else "dealer"
    return "adu" if author.lower().strip(" '\"") in staff else "dealer"


# --------------------------------------------------------------------------- #
# Splitting one-line email text into its messages                              #
# --------------------------------------------------------------------------- #

# Outlook: "From: Andrew Domshick Sent: Friday, April 10, 2026 4:52 PMTo: ..."
OUTLOOK = re.compile(r"From:\s*(?P<author>.{1,120}?)\s*(?:Sent|Date):\s*(?P<date>.{6,60}?[AP]M)")
# Gmail / iPhone: "On Mon, Aug 18, 2025 at 9:22 AM Jane Doe <ja***@de**ft.com> wrote:"
GMAIL = re.compile(r"On (?P<date>[A-Z][^<]{5,60}?[AP]M),?\s*(?P<author>[^<]{0,60}<[^>]{3,80}>)\s*wrote:")

# After an Outlook header, skip the To:/Cc:/Subject: part up to where the body starts.
# The subject has no end marker, so a short subject can leak into the body - harmless.
OUTLOOK_REST = re.compile(r"^\s*To:.*?Subject:\s*", re.S)


REPLY_PREFIX = re.compile(r"^\s*((re|fw|fwd)\s*:\s*)+", re.I)


def split_messages(text, sender, sent_on, subject, staff):
    """Cut an email into messages: the newest on top, quoted ones after it."""
    base_subject = REPLY_PREFIX.sub("", subject)
    leaked_subject = re.compile(rf"^\s*((re|fw|fwd)\s*:\s*)*{re.escape(base_subject)}\s*", re.I)
    headers = sorted(list(OUTLOOK.finditer(text)) + list(GMAIL.finditer(text)),
                     key=lambda m: m.start())
    messages = [{"author": sender, "date": sent_on,
                 "text": text[:headers[0].start()] if headers else text}]
    for header, nxt in zip(headers, headers[1:] + [None]):
        body = text[header.end():nxt.start() if nxt else len(text)]
        if header.re is OUTLOOK:
            body = OUTLOOK_REST.sub("", body, count=1)
            if base_subject:
                body = leaked_subject.sub("", body, count=1)
        messages.append({"author": header["author"].strip(), "date": header["date"], "text": body})

    for position, m in enumerate(messages):
        m["position"] = position
        m["role"] = role_of(m["author"], staff)
        m["text"] = strip_boilerplate(m["text"])
    return [m for m in messages if m["text"]]


# --------------------------------------------------------------------------- #
# Boilerplate - with no line breaks we remove known patterns instead of lines  #
# --------------------------------------------------------------------------- #

FOOTER_START = re.compile(
    r"((ARMATUS\s+)?CONFIDENTIALITY NOTICE|DISCLAIMER:|This e-?mail (transmission|and any|message)"
    r"|Sent from my (iPhone|iPad|Galaxy)|Get Outlook for (iOS|Android))", re.I)

BOILERPLATE = [
    r"(Direct|Cell|Mobile|Office|Phone|Fax|Tel)\s*:\s*[\d*() .-]{7,}",   # phone lines
    r"\(?[\d*]{3}\)?[\s.-]?[\d*]{3}[\s.-]?\d{4}",                        # bare phones
    MASKED_ADDRESS.pattern,                                              # email addresses
    r"(https?://|www\.)\S+",                                             # links
    r"50 Schilling Road,?\s*Suite 200\s*Hunt Valley,?\s*MD 21031",       # ADU office
    r"(Please )?CLICK HERE [^!]{0,80}!",                                 # marketing links
    r"6,000\+ Happy Dealerships - and counting!",
    r"\*?Siri (is )?frequently used, I apologize for any typos\.",
    r"_{5,}|-{5,}",                                                      # divider lines
    r"\[IMG\]",
]
BOILERPLATE_RE = re.compile("|".join(f"(?:{p})" for p in BOILERPLATE), re.I)


def strip_boilerplate(text):
    footer = FOOTER_START.search(text)
    if footer:
        text = text[:footer.start()]
    text = BOILERPLATE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" |,;")


# --------------------------------------------------------------------------- #
# Stores: is an activity about one store, or the whole group?                  #
# --------------------------------------------------------------------------- #

# "Related Companies" in the company record lists the group's stores:
#   "Maple Honda (#100000001); Maple Kia (#100000002); ..."
RELATED_COMPANY = re.compile(r"\s*(.+?)\s*\(#(\d+)\)\s*$")

# Related companies that are groups or management companies, not stores.
SUB_GROUP = re.compile(r"\b(auto group|automotive group|management|dealer group)\b", re.I)

# Shorthand used in project names, e.g. "Airport CDJR", "Maple BGMC Cadillac".
ABBREVIATIONS = {"cdjr": "chrysler dodge jeep ram", "cdj": "chrysler dodge jeep",
                 "bgmc": "buick gmc", "vw": "volkswagen", "chevy": "chevrolet"}
IGNORED_WORDS = {"of", "the", "and", "nlop", "ca", "mo"}

# An activity naming at least this share of the group's stores is about the group.
GROUP_SHARE = 0.75


def name_tokens(name):
    """'NLOP_Airport CDJR' -> {'airport', 'chrysler', 'dodge', 'jeep', 'ram'}"""
    words = re.findall(r"[a-z0-9]+", name.lower().replace("_", " "))
    words = " ".join(ABBREVIATIONS.get(w, w) for w in words).split()
    return {w for w in words if w not in IGNORED_WORDS}


def store_directory(info):
    """The company's stores as [{'id', 'name', 'active'}]. A solo store is its own store."""
    if info["Company Type"] == "Solo Store":
        return [{"id": info["Company ID"], "name": info["Company Name"], "active": True}]
    stores = []
    for part in value(info.get("Related Companies")).split(";"):
        found = RELATED_COMPANY.match(part)
        if found and not SUB_GROUP.search(found[1]):
            name = found[1]
            stores.append({"id": int(found[2]), "name": name,
                           "active": not name.upper().startswith("NLOP")})
    return stores


def match_store(name, directory):
    """Best store in the directory for a name as written in an activity, or None.

    Handles project names ("Maple Subaru: Parts R6"), group prefixes
    ("MapleAG_Maple Honda"), NLOP_ markers and shorthand ("Airport CDJR").
    """
    name = name.split(":")[0]                          # drop the project part
    tokens = name_tokens(name)
    if not tokens:
        return None
    best, best_score = None, 0.0
    for store in directory:
        store_tokens = name_tokens(store["name"])
        score = len(tokens & store_tokens) / len(tokens | store_tokens)
        if score > best_score:
            best, best_score = store, score
    return best if best_score >= 0.7 else None


def stores_named_in(text, directory):
    """Stores whose full name appears in free text, e.g. an email subject like
    "RE: Project Restart Reminder - Maple Honda Parts"."""
    words = name_tokens(text)
    return [s for s in directory if len(name_tokens(s["name"])) >= 2 and name_tokens(s["name"]) <= words]


def store_level(a, text, info, directory, subject=""):
    """Work out which store(s) an activity is about.

    Looks at the activity's Group and Stores values (and an email's subject),
    matched against the store list. Pieces that match no store - contact names,
    or the details text the export sometimes copies into Stores - are ignored.
    Returns {"level": "store"|"group", "store_ids": [...], "stores": "names"}.
    """
    group_name = info["Company Name"]
    if info["Company Type"] == "Solo Store":
        return {"level": "store", "store_ids": [info["Company ID"]], "stores": group_name}

    names = [value(a["Group"])] + value(a["Stores"]).split(", ")
    matched = {}
    for name in names:
        store = match_store(name, directory) if name and name != group_name else None
        if store:
            matched[store["id"]] = store["name"]
    for store in stores_named_in(subject, directory) if subject else []:
        matched[store["id"]] = store["name"]

    all_stores = "ALL STORES" in text.upper()
    if all_stores or not matched or (len(matched) > 1 and
                                     len(matched) >= GROUP_SHARE * len(directory)):
        return {"level": "group", "store_ids": [], "stores": group_name}
    return {"level": "store", "store_ids": sorted(matched),
            "stores": ", ".join(matched[i] for i in sorted(matched))}


# --------------------------------------------------------------------------- #
# Building the timeline                                                        #
# --------------------------------------------------------------------------- #

def from_activity(a, info, directory):
    text = "\n".join(dict.fromkeys(p for p in (value(a["Title"]), value(a["Details"])) if p))
    if not text:
        return None
    return {
        "ref": f"A{value(a['Activity ID']) or fingerprint(a['Date'], text)}",
        "date": parse_date(a["Date"]),
        "source": "activity",
        "kind": KINDS.get(a["Type"], a["Type"].lower()),
        "role": "adu",
        "author": value(a["Assigned To"]),
        **store_level(a, text, info, directory),
        "subject": None,
        "text": redact(text),
    }


def message_key(role, text):
    """Same role + same opening words = same message.

    Not the author: a message is signed with an address on top of one email and
    just a name when quoted in another. Only the opening words: the tail differs
    between copies (signature leftovers), and a truncated ~200-char preview
    still matches the full copy quoted in a later email.
    """
    opening = re.sub(r"[^a-z0-9]", "", text.lower())[:150]
    return fingerprint(role, opening)


def from_emails(emails, staff, info, directory):
    """One activity per unique message across all of a company's emails."""
    newest_first = sorted(emails, key=lambda a: a["Date"], reverse=True)
    by_print = {}
    for rank, e in enumerate(newest_first):
        truncated = rank >= PORTAL_FULL_EMAIL_LIMIT
        text = value(e["Message"])
        for m in split_messages(text, value(e["From"]), e["Date"], value(e["Details"]), staff):
            key = message_key(m["role"], m["text"])
            date = parse_date(m["date"])
            estimated = date is None
            if estimated:
                date = parse_date(e["Date"])
            activity = by_print.get(key)
            if activity is None:
                by_print[key] = {
                    "ref": f"E{key}",
                    "date": date,
                    "date_estimated": estimated,
                    "source": "email",
                    "kind": "email",
                    "role": m["role"],
                    "author": m["author"],
                    **store_level(e, value(e["Message"]), info, directory, value(e["Details"])),
                    "subject": value(e["Details"]),     # the portal puts the subject here
                    "text": redact(m["text"]),
                    "truncated": truncated,
                }
            else:
                if date and not estimated and (activity["date_estimated"] or date < activity["date"]):
                    activity["date"], activity["date_estimated"] = date, False
                if len(m["text"]) > len(activity["text"]):      # keep the fullest copy
                    activity["text"], activity["truncated"] = redact(m["text"]), truncated
    return list(by_print.values())


def build_company(company, staff):
    info = company["Company"][0]
    raw = company["Activities"]
    directory = store_directory(info)
    activities = [x for a in raw if a["Type"] != "Email"
                  for x in [from_activity(a, info, directory)] if x]
    activities += from_emails([a for a in raw if a["Type"] == "Email"], staff, info, directory)
    activities.sort(key=lambda x: (x["date"] is None, x["date"] or ""))
    # Short ids - 1, 2, 3 ... in date order - that a model can copy without
    # mistakes. "ref" keeps the long original id for tracing back to the source.
    activities = [{"id": n, **a} for n, a in enumerate(activities, start=1)]
    return {
        "company_id": info["Company ID"],
        "company_name": info["Company Name"],
        "company_type": info["Company Type"],
        "number_of_stores": info["Number Of Stores"],
        "store_list": directory,
        "count": len(activities),
        "activities": activities,
    }


if __name__ == "__main__":
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    companies = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(LOCAL_DIR.glob("company_*.json"))]
    staff = adu_names(companies)

    for company in companies:
        t = build_company(company, staff)
        out = TIMELINE_DIR / f"company_{t['company_id']}.json"
        out.write_text(json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")

        acts = t["activities"]
        n = lambda **kw: sum(all(x.get(k) == v for k, v in kw.items()) for x in acts)
        dates = [x["date"] for x in acts if x["date"]]
        print(f"  {t['company_name'][:28]:<28} {len(acts):>4} activities  "
              f"(non-email {n(source='activity'):>3}, email ADU {n(source='email', role='adu'):>3}, "
              f"email dealer {n(source='email', role='dealer'):>3}, truncated {n(truncated=True):>3})  "
              f"{dates[0][:10]} -> {dates[-1][:10]}")
        print(f"  {'':<28} level: store {n(level='store'):>4}, group {n(level='group'):>4}")
