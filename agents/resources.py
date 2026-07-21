"""Read the actual content behind a link (or local file) so it can be ingested.

Supports: GitHub repos (README), YouTube videos (transcript), arXiv papers
(abstract), PDFs (linked URL or local file), and a best-effort generic web page.
Every fetcher fails soft: on any error it returns "" so ingestion can continue.
"""
import io
import os
import re

import requests

# --- link discovery -------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
# titles/snippets often show a repo as "owner/repo: description"
_REPO_RE = re.compile(r"\b([A-Za-z0-9][\w.-]+)/([A-Za-z0-9][\w.-]+):")


def extract_links(text):
    """Return a de-duplicated list of resource URLs found in free text."""
    links = []
    seen = set()

    def add(u):
        u = u.rstrip(".,);]'\"")
        if u and u not in seen:
            seen.add(u)
            links.append(u)

    for u in _URL_RE.findall(text or ""):
        add(u)
    for owner, repo in _REPO_RE.findall(text or ""):
        add(f"https://github.com/{owner}/{repo}")
    return links


# --- per-type fetchers ----------------------------------------------------

def _github_readme(url):
    m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", url)
    if not m:
        return ""
    owner, repo = m.group(1), m.group(2).replace(".git", "").rstrip("/")
    headers = {"Accept": "application/vnd.github.raw+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers=headers,
            timeout=20,
        )
        if r.status_code == 200:
            return r.text
    except requests.RequestException:
        pass
    return ""


def _youtube_transcript(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if not m:
        return ""
    vid = m.group(1)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):  # newer library API
            parts = api.fetch(vid)
            parts = getattr(parts, "snippets", parts)
        else:  # classic library API
            parts = YouTubeTranscriptApi.get_transcript(vid)
        out = [p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
               for p in parts]
        return " ".join(t for t in out if t)
    except Exception:
        return ""


def _arxiv_abstract(url):
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?/?$", url)
    if not m:
        return ""
    try:
        r = requests.get(
            f"http://export.arxiv.org/api/query?id_list={m.group(1)}", timeout=20
        )
        title = re.search(r"<entry>.*?<title>(.*?)</title>", r.text, re.S)
        summary = re.search(r"<summary>(.*?)</summary>", r.text, re.S)
        out = []
        if title:
            out.append(re.sub(r"\s+", " ", title.group(1)).strip())
        if summary:
            out.append(re.sub(r"\s+", " ", summary.group(1)).strip())
        return "\n".join(out)
    except requests.RequestException:
        return ""


def _pdf_text(data):
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _pdf_from_url(url):
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return _pdf_text(r.content)
    except requests.RequestException:
        pass
    return ""


def _generic_web(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "html" in r.headers.get("Content-Type", ""):
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S)
            html = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", html).strip()
    except requests.RequestException:
        pass
    return ""


# --- classification / redirects -------------------------------------------

_SUPPORTED_HOSTS = ("github.com", "youtube.com", "youtu.be", "arxiv.org")
_SHORTENERS = ("lnkd.in", "bit.ly", "buff.ly", "t.co", "ow.ly", "tinyurl.com")


def is_supported_resource(url):
    """True only for link types we can reliably read (github/youtube/arxiv/pdf)."""
    low = (url or "").lower()
    if any(h in low for h in _SUPPORTED_HOSTS):
        return True
    return low.split("?")[0].endswith(".pdf")


def is_shortlink(url):
    return any(s in (url or "").lower() for s in _SHORTENERS)


def resolve_redirect(url):
    """Follow shortlink redirects (e.g. lnkd.in) to the real destination URL."""
    try:
        r = requests.head(
            url, allow_redirects=True, timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        return r.url or url
    except requests.RequestException:
        return url


# --- dispatcher -----------------------------------------------------------

def fetch_resource(url_or_path):
    """Detect the resource type and return (kind, text). text is "" on failure."""
    src = (url_or_path or "").strip()
    if not src:
        return ("", "")

    if os.path.isfile(src):
        if src.lower().endswith(".pdf"):
            try:
                with open(src, "rb") as f:
                    return ("pdf", _pdf_text(f.read()))
            except OSError:
                return ("pdf", "")
        try:
            with open(src, encoding="utf-8", errors="ignore") as f:
                return ("file", f.read())
        except OSError:
            return ("file", "")

    low = src.lower()
    if "github.com/" in low:
        return ("github", _github_readme(src))
    if "youtube.com/" in low or "youtu.be/" in low:
        return ("youtube", _youtube_transcript(src))
    if "arxiv.org/" in low:
        return ("arxiv", _arxiv_abstract(src))
    if low.split("?")[0].endswith(".pdf"):
        return ("pdf", _pdf_from_url(src))
    return ("web", _generic_web(src))
