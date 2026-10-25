"""
Verify Muna - paste a message, see what trusted sources say about it.

The system does NOT decide whether a claim is true or false.
It only searches a curated list of trusted sites and shows what it finds.
"""

import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import yake
from ddgs import DDGS
from flask import Flask, render_template, request
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DetectorFactory.seed = 0  # makes language detection consistent

app = Flask(__name__)

# domain -> (display name, source type). Edit this list to add or remove sources.
TRUSTED_SOURCES = {
    "pagasa.dost.gov.ph": ("PAGASA", "Government"),
    "phivolcs.dost.gov.ph": ("PHIVOLCS", "Government"),
    "comelec.gov.ph": ("COMELEC", "Government"),
    "doh.gov.ph": ("Department of Health", "Government"),
    "pna.gov.ph": ("Philippine News Agency", "Government"),
    "verafiles.org": ("VERA Files", "Fact-checker"),
    "tsek.ph": ("Tsek.ph", "Fact-checker"),
    "factcheck.afp.com": ("AFP Fact Check", "Fact-checker"),
    "rappler.com": ("Rappler", "News"),
    "inquirer.net": ("Inquirer", "News"),
    "gmanetwork.com": ("GMA News", "News"),
    "abs-cbn.com": ("ABS-CBN News", "News"),
    "philstar.com": ("Philstar", "News"),
    "mb.com.ph": ("Manila Bulletin", "News"),
    "manilatimes.net": ("The Manila Times", "News"),
    # International outlets (add or remove to suit your project)
    "reuters.com": ("Reuters", "International"),
    "apnews.com": ("Associated Press", "International"),
    "bbc.com": ("BBC News", "International"),
    "bbc.co.uk": ("BBC News", "International"),
    "cnn.com": ("CNN", "International"),
    "theguardian.com": ("The Guardian", "International"),
    "aljazeera.com": ("Al Jazeera", "International"),
    "forbes.com": ("Forbes", "International"),
}

# Small list of common Filipino filler words so they don't become "keywords"
FIL_STOPWORDS = {
    "ang", "ng", "sa", "na", "at", "ay", "mga", "ako", "ka", "ko", "mo", "niya",
    "kami", "tayo", "sila", "ito", "iyan", "iyon", "yung", "daw", "raw", "po",
    "ba", "din", "rin", "lang", "naman", "pa", "pag", "kung", "para", "dahil",
    "may", "mayroon", "wala", "ni", "kay", "si", "sina", "nang", "pero", "o",
    # chain-message filler that should not drive the search
    "breaking", "share", "shared", "forward", "forwarded", "viral", "now",
    "news", "urgent", "ipasa", "ipamahagi", "pakishare", "pls", "please",
}

MIN_LENGTH = 15
MAX_LENGTH = 2000


def detect_language(text):
    try:
        code = detect(text.lower())
    except Exception:
        return "Unknown"
    if code == "tl":
        return "Filipino / Tagalog"
    if code == "en":
        return "English"
    # Short or mixed Taglish text is often misdetected as another language
    return "English / Filipino (mixed)"


def get_keywords(text, n=6):
    """Extract the main keywords/keyphrases from the pasted text using YAKE."""
    extractor = yake.KeywordExtractor(lan="en", n=2, top=n * 2)
    cleaned = []
    for phrase, _score in extractor.extract_keywords(text):
        words = [w for w in phrase.lower().split() if w not in FIL_STOPWORDS]
        phrase = " ".join(words)
        if phrase and phrase not in cleaned:
            cleaned.append(phrase)
    return cleaned[:n]


def warning_flags(text):
    """Simple warning signs often seen in misinformation. These are hints, not proof."""
    flags = []

    letters = [c for c in text if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        flags.append("Much of the text is in ALL CAPS.")

    if text.count("!") >= 3:
        flags.append("Uses many exclamation marks.")

    if re.search(r"\b(share|i-share|ishare|ipasa|ipamahagi|forward|viral)\b", text, re.I):
        flags.append("Asks you to share or forward it.")

    if not re.search(r"https?://|\b(according to|ayon (kay|sa)|source|sanggunian)\b", text, re.I):
        flags.append("Does not name a source or include a link.")

    return flags


MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}

# Places where news sites usually store the publish date in the page's HTML
DATE_PATTERNS = [
    r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)',
    r'content=["\']([^"\']+)["\'][^>]*property=["\']article:published_time["\']',
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'name=["\'](?:pubdate|publishdate|date|DC\.date\.issued)["\'][^>]*content=["\']([^"\']+)',
    r'<time[^>]*datetime=["\']([^"\']+)',
]


def date_from_url(url):
    """Some sites put the publish date in the URL, e.g. /2025/11/04/."""
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", url)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""


def date_from_text(text):
    """Find a date written like 'Oct 12, 2026' or '2026-10-12' inside some text."""
    m = re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", text)
    if m:
        return m.group(0)
    m = re.search(r"\b([A-Za-z]{3})[a-z]*\.? (\d{1,2}),? (20\d{2})\b", text)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return ""


def date_from_page(url):
    """Open the article and read its publish date from the page's metadata."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            html = resp.read(400000).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, html, re.I)
        if m:
            found = date_from_text(m.group(1))
            if found:
                return found
    return ""


def find_dates(items):
    """Fill in each result's date (URL first, then snippet, then the page itself)."""
    for it in items:
        it["date"] = (
            date_from_url(it["link"])
            or date_from_text(f"{it['title']} {it['snippet']}")
        )
    missing = [it for it in items if not it["date"]]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for it, found in zip(missing, pool.map(lambda i: date_from_page(i["link"]), missing)):
                it["date"] = found
    return items


def pretty_date(iso):
    y, m, d = (int(x) for x in iso.split("-"))
    return date(y, m, d).strftime("%B %d, %Y").replace(" 0", " ")


def source_info(url):
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
    for domain, info in TRUSTED_SOURCES.items():
        if host == domain or host.endswith("." + domain):
            return info
    return (host, "Other")


def _ddg_text(query, max_results, timelimit=None):
    """Run a DuckDuckGo text search (no API key needed). Returns [] when nothing is found."""
    try:
        return (
            DDGS().text(
                query, region="ph-en", max_results=max_results, timelimit=timelimit
            )
            or []
        )
    except Exception as exc:
        if "no results" in str(exc).lower():
            return []
        raise


def search_trusted(query, timelimit=None):
    """Search the web, keeping only results from the trusted domains."""
    sites = " OR ".join(f"site:{d}" for d in TRUSTED_SOURCES)
    raw = _ddg_text(f"{query} ({sites})", 20, timelimit)

    if not raw:
        # Fallback: search normally, then keep only trusted domains
        raw = [
            r for r in _ddg_text(f"{query} Philippines", 30, timelimit)
            if source_info(r.get("href", ""))[1] != "Other"
        ]

    items = []
    for r in raw:
        name, kind = source_info(r.get("href", ""))
        if kind == "Other":
            continue
        items.append(
            {
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": (r.get("body") or "").replace("\n", " "),
                "source": name,
                "type": kind,
                "date": "",
            }
        )
    return items


def rank_results(text, items):
    """Rank results by how similar their title + snippet are to the pasted text."""
    if not items:
        return items
    docs = [text] + [f"{i['title']} {i['snippet']}" for i in items]
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(docs)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    for item, sim in zip(items, sims):
        score = round(float(sim) * 100)
        item["score"] = score
        # Thresholds are rough; adjust them after testing with real examples.
        if score >= 30:
            item["match"] = "Strong match"
        elif score >= 12:
            item["match"] = "Possible match"
        else:
            item["match"] = "Weak match"
    return sorted(items, key=lambda i: i["score"], reverse=True)


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {
        "text": "",
        "searched": False,
        "keywords": [],
        "results": [],
        "flags": [],
        "language": None,
        "period": "y",
        "hidden": 0,
        "weak_hidden": 0,
        "year": date.today().year,
        "error": None,
    }

    if request.method == "POST":
        text = request.form.get("claim", "").strip()
        ctx["text"] = text
        period = request.form.get("period", "y")
        ctx["period"] = period if period in ("w", "m", "y") else "y"

        if len(text) < MIN_LENGTH:
            ctx["error"] = "Paste a longer message so the system has something to check."
        elif len(text) > MAX_LENGTH:
            ctx["error"] = f"That is too long. Keep it under {MAX_LENGTH} characters."
        else:
            ctx["searched"] = True
            ctx["language"] = detect_language(text)
            ctx["flags"] = warning_flags(text)
            ctx["keywords"] = get_keywords(text)

            try:
                items = search_trusted(
                    " ".join(ctx["keywords"]), ctx["period"]
                )
                items = find_dates(items)
                this_year = date.today().year
                current = [i for i in items if i["date"].startswith(str(this_year))]
                ctx["hidden"] = len(items) - len(current)
                ctx["year"] = this_year
                for i in current:
                    i["date_text"] = pretty_date(i["date"])
                ranked = rank_results(text, current)
                # Weak matches are usually unrelated articles, so leave them out
                ctx["weak_hidden"] = sum(i["match"] == "Weak match" for i in ranked)
                ranked = [i for i in ranked if i["match"] != "Weak match"]
                # Most recent first; ties are broken by how well they match
                ctx["results"] = sorted(
                    ranked, key=lambda i: (i["date"], i["score"]), reverse=True
                )
            except Exception:
                ctx["error"] = (
                    "The search could not be completed. Check your internet "
                    "connection and try again in a minute."
                )
                ctx["searched"] = False

    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
