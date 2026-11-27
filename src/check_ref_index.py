"""Ask the public indexes whether the refuted baseline study was ever published.

Why this module exists
----------------------
``masaba2023`` is the study this paper refutes: an author-held technical draft
that reported precision, recall and ROC-AUC all equal to 1.0 on a Ugandan
weather record. Because we contradict it, an editor or reviewer will try to look
it up. If it cannot be found we have to say so precisely and show the search,
not assert the absence.

Queries run here (standard library only, no API keys):
  Crossref          query.bibliographic / query.title / query.author
  OpenAlex          title.search filter, free-text search, author work lists
  arXiv             Atom API, title and all-field probes
  Semantic Scholar  best effort (the public endpoint rate-limits)

Writes results/masaba2023_index_check.json: every query URL, HTTP status, hit
count, the records whose titles were compared against the target, and the
verdict.

Failure policy: a request that does not answer is recorded as answered=false and
is never counted as evidence of absence. The verdict "no indexed record" is
emitted only when Crossref, OpenAlex and arXiv all answered; otherwise the
script exits non-zero, so a network failure cannot masquerade as a result.

Usage:  python src/check_ref_index.py
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

# The draft's own title page, as supplied by the author.
TITLE = "Evaluation of Anomaly Detection Techniques Using Uganda Weather Dataset"
AUTHOR = "Jeremiah Masaba"
YEAR = "2023"
# Date the committed artifact was generated; a re-run records its own date.
CHECKED_ON = "2026-11-27"
OUT = C.RESULTS / "masaba2023_index_check.json"
# mailto puts OpenAlex/Crossref queries in the polite pool and is public anyway.
CONTACT = "masabjericho@gmail.com"
UA = {"User-Agent": "ucasi-reference-check/1.0 (mailto:%s)" % CONTACT}
REQUIRED = ("crossref", "openalex", "arxiv")
MAX_RECORDS = 6


def norm(s):
    """Lower-case, collapse punctuation/whitespace: a comparison key for titles."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


TARGET = norm(TITLE)
TARGET_WORDS = set(TARGET.split())
# Titles in this area are built from generic tokens ("anomaly", "detection",
# "techniques", "dataset"); only these two make a hit about this record.
DISTINCTIVE = ("uganda", "weather")
CANDIDATE_LEVELS = ("exact", "substring", "strong_overlap")


def match_level(candidate):
    """Graded similarity between a found title and the target title.

    weak_overlap exists because dozens of unrelated papers share the generic
    wording; it is recorded for transparency but is not a candidate match.
    """
    c = norm(candidate)
    if not c:
        return None
    if c == TARGET:
        return "exact"
    if TARGET in c or c in TARGET:
        return "substring"
    words = set(c.split())
    coverage = len(words & TARGET_WORDS) / max(len(TARGET_WORDS), 1)
    if coverage >= 0.6:
        distinctive = sum(1 for w in DISTINCTIVE if w in words)
        if distinctive == len(DISTINCTIVE) and "anomal" in c:
            return "strong_overlap"
        return "weak_overlap"
    return None


def is_candidate(level):
    return level in CANDIDATE_LEVELS


def fetch(url, tries=3, pause=4.0):
    """Return (status, payload, error). Never raises; the caller records."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.status, r.read().decode("utf-8", "replace"), None
        except urllib.error.HTTPError as exc:
            last = "HTTP %s" % exc.code
            if exc.code in (429, 500, 502, 503):
                time.sleep(pause * (attempt + 1))
                continue
            return exc.code, None, last
        except Exception as exc:                      # DNS, timeout, TLS
            last = repr(exc)
            time.sleep(pause)
    return None, None, last


def rank(records):
    """Title matches first, then enough near-misses to show what was inspected."""
    strong = [r for r in records if r.get("title_match")]
    rest = [r for r in records if not r.get("title_match")]
    return strong[:MAX_RECORDS] + rest[:max(0, MAX_RECORDS - len(strong))]


def probe(index, label, url, parse):
    """Run one query, keep only what a reader needs to re-check it."""
    status, body, err = fetch(url)
    entry = {"index": index, "label": label, "url": url,
             "http_status": status, "answered": body is not None}
    if err:
        entry["error"] = err
    if body is None:
        entry.update(n_hits=0, n_target_matches=0, records=[])
        return entry, None
    try:
        hits, records = parse(body)
    except Exception as exc:
        entry["error"] = "parse failed: %r" % (exc,)
        entry.update(n_hits=0, n_target_matches=0, records=[])
        return entry, None
    levels = [r["title_match"] for r in records if r.get("title_match")]
    candidates = sum(1 for r in records if is_candidate(r.get("title_match")))
    entry.update(n_hits=hits, n_target_matches=candidates,
                 match_levels=sorted(set(levels)), records=records)
    return entry, levels


# -------------------------------------------------------------------- indexes
def crossref(params):
    q = urllib.parse.urlencode(dict(params, rows=MAX_RECORDS))
    return probe("crossref", "%s=%s" % (list(params)[0], list(params.values())[0]),
                 "https://api.crossref.org/works?" + q, _parse_crossref)


def _parse_crossref(body):
    items = json.loads(body).get("message", {}).get("items", [])
    records = []
    for it in items[:10]:
        title = (it.get("title") or [""])[0]
        records.append({
            "title": title[:160],
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (it.get("container-title") or [""])[0][:80],
            "type": it.get("type"),
            "doi": it.get("DOI"),
            "authors": ", ".join(a.get("family", "?")
                                 for a in it.get("author", [])[:5]),
            "title_match": match_level(title),
        })
    hits = len(items)
    return hits, rank(records)


def openalex(params, label, url):
    return probe("openalex", label, "https://api.openalex.org/%s" % url,
                 lambda b: _parse_openalex(b, params))


def _parse_openalex(body, kind):
    data = json.loads(body)
    if kind == "authors":
        results = data.get("results", [])
        records = []
        for w in results[:MAX_RECORDS]:
            inst = (w.get("last_known_institutions") or [{}])[0].get("display_name")
            records.append({"author": w.get("display_name"),
                            "orcid": (w.get("orcid") or "")[-16:],
                            "works": w.get("works_count"),
                            "affiliation": inst,
                            "id": w.get("id")})
        return data.get("meta", {}).get("count", len(results)), records
    works = data.get("results", [])
    records = []
    for w in works[:10]:
        title = w.get("display_name") or ""
        loc = w.get("primary_location") or {}
        records.append({
            "title": title[:160],
            "year": w.get("publication_year"),
            "venue": ((loc.get("source") or {}).get("display_name") or "")[:80],
            "type": w.get("type"),
            "doi": w.get("doi"),
            "authors": ", ".join((a.get("author") or {}).get("display_name", "?")
                                 for a in w.get("authorships", [])[:5]),
            "title_match": match_level(title),
        })
    hits = data.get("meta", {}).get("count", len(works))
    return hits, rank(records)


def arxiv(query, label):
    url = ("http://export.arxiv.org/api/query?search_query=%s&max_results=%d"
           % (urllib.parse.quote(query), MAX_RECORDS))
    return probe("arxiv", label, url, _parse_arxiv)


def _parse_arxiv(body):
    entries = body.split("<entry>")[1:]
    total = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", body)
    records = []
    for e in entries:
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        aid = re.search(r"<id>(.*?)</id>", e)
        when = re.search(r"<published>(\d{4})", e)
        t = re.sub(r"\s+", " ", title.group(1)).strip() if title else ""
        records.append({"title": t[:160],
                        "year": when.group(1) if when else None,
                        "id": aid.group(1) if aid else None,
                        "title_match": match_level(t)})
    return int(total.group(1)) if total else len(entries), records


def semantic_scholar():
    q = urllib.parse.urlencode({"query": TITLE, "limit": MAX_RECORDS,
                                "fields": "title,year,venue,externalIds,authors"})
    return probe("semanticscholar", "paper search",
                 "https://api.semanticscholar.org/graph/v1/paper/search?" + q,
                 _parse_s2)


def _parse_s2(body):
    data = json.loads(body).get("data", [])
    records = []
    for p in data[:10]:
        title = p.get("title") or ""
        records.append({
            "title": title[:160], "year": p.get("year"),
            "venue": (p.get("venue") or "")[:80],
            "ids": p.get("externalIds"),
            "authors": ", ".join(a.get("name", "?") for a in p.get("authors", [])[:5]),
            "title_match": match_level(title),
        })
    hits = len(data)
    return hits, rank(records)


# --------------------------------------------------------------------- author
def openalex_author_works():
    """List what OpenAlex holds under this author's name; look for the title."""
    url = ("works?" + urllib.parse.urlencode(
        {"filter": "raw_author_name.search:%s" % AUTHOR,
         "per-page": 100, "select": "display_name,publication_year,type,doi",
         "mailto": CONTACT}))
    status, body, err = fetch("https://api.openalex.org/" + url)
    out = {"index": "openalex", "label": "works by raw_author_name=%s" % AUTHOR,
           "url": "https://api.openalex.org/" + url, "http_status": status,
           "answered": body is not None}
    if body is not None:
        works = json.loads(body).get("results", [])
        hits = [w for w in works if match_level(w.get("display_name", ""))]
        counted = [w for w in works if is_candidate(match_level(w.get("display_name", "")))]
        out.update(n_hits=len(works), n_target_matches=len(counted),
                   match_levels=sorted({match_level(h["display_name"])
                                         for h in hits}),
                   records=[{"title": (w.get("display_name") or "")[:160],
                             "year": w.get("publication_year"),
                             "doi": w.get("doi"),
                             "title_match": match_level(w.get("display_name", ""))}
                            for w in hits[:MAX_RECORDS]])
    else:
        out["error"] = err
        out.update(n_hits=0, n_target_matches=0, records=[])
    return out


def openalex_nearest():
    """Closest indexed Ugandan weather-data records, and who owns them.

    This is the positive control for the negative result: if the query machinery
    works, searching the distinctive half of the title must still return the
    real Ugandan weather records. Their authors are recorded so a reader can see
    that none of them is the draft under examination.
    """
    url = ("works?" + urllib.parse.urlencode(
        {"filter": "title.search:uganda weather", "per-page": 25,
         "sort": "publication_year:desc",
         "select": "id,display_name,publication_year,type,doi,primary_location,"
                   "authorships", "mailto": CONTACT}))
    status, body, err = fetch("https://api.openalex.org/" + url)
    if body is None:
        return {"answered": False, "http_status": status, "error": err, "records": []}
    payload = json.loads(body)
    records = []
    for w in payload.get("results", []):
        names = [((a.get("author") or {}).get("display_name") or "")
                 for a in w.get("authorships", [])]
        words = set(norm(w.get("display_name", "")).split())
        records.append({
            "title": (w.get("display_name") or "")[:160],
            "year": w.get("publication_year"),
            "type": w.get("type"),
            "doi": w.get("doi"),
            "venue": (((w.get("primary_location") or {}).get("source"))
                      or {}).get("display_name"),
            "authors": ", ".join(names)[:160],
            "author_matches_target": any(
                "masaba" in n.lower() and "jeremiah" in n.lower() for n in names),
            "title_match": match_level(w.get("display_name", "")),
            "title_word_overlap": round(
                len(words & TARGET_WORDS) / max(len(TARGET_WORDS), 1), 3),
        })
    records.sort(key=lambda r: (not r["author_matches_target"],
                                not is_candidate(r["title_match"]),
                                -r["title_word_overlap"]))
    shown = ([r for r in records if r["author_matches_target"]
              or is_candidate(r["title_match"])] or records[:4])
    return {"answered": True, "http_status": status,
            "index_count": payload.get("meta", {}).get("count"),
            "n_scanned": len(records), "records": shown}


def main():
    queries = []
    queries.append(crossref({"query.bibliographic": TITLE})[0])
    queries.append(crossref({"query.title": TITLE})[0])
    queries.append(crossref({"query.bibliographic": "Uganda weather dataset",
                             "query.author": "Masaba"})[0])
    queries.append(openalex(None, "title.search", "works?" + urllib.parse.urlencode(
        {"filter": "title.search:%s" % TITLE, "per-page": MAX_RECORDS,
         "mailto": CONTACT}))[0])
    queries.append(openalex(None, "free-text search", "works?" + urllib.parse.urlencode(
        {"search": TITLE, "per-page": MAX_RECORDS, "mailto": CONTACT}))[0])
    queries.append(openalex(None, "author search: %s" % AUTHOR, "authors?"
                           + urllib.parse.urlencode(
                               {"search": AUTHOR, "per-page": MAX_RECORDS,
                                "mailto": CONTACT}))[0])
    queries.append(openalex_author_works())
    queries.append(arxiv('ti:"anomaly detection" AND abs:Uganda',
                         "title/abstract probe")[0])
    queries.append(arxiv('all:"Uganda weather dataset"', "all-field phrase")[0])
    queries.append(semantic_scholar()[0])

    nearest = openalex_nearest()

    by_index = {}
    for q in queries:
        by_index.setdefault(q["index"], []).append(q)

    unanswered = [q["index"] for q in queries
                  if not q["answered"] and q["index"] in REQUIRED]
    matched = [q for q in queries if q.get("n_target_matches", 0) > 0]
    verdict = {
        "no_indexed_record": (not unanswered) and not matched,
        "required_indexes_unanswered": sorted(set(unanswered)),
        "queries_with_a_title_match": [
            {"index": q["index"], "label": q["label"], "levels": q.get("match_levels")}
            for q in matched],
    }
    if unanswered:
        verdict["no_indexed_record"] = None
    artifact = {
        "purpose": ("Checks whether the unpublished baseline draft refuted in this "
                    "paper (masaba2023) is recorded as published in any index."),
        "target": {"title": TITLE, "author": AUTHOR, "year": YEAR,
                   "normalised_title": TARGET},
        "checked_on": CHECKED_ON,
        "method": ("src/check_ref_index.py; stdlib HTTP GET against public APIs; "
                   "title comparison is case/punctuation-insensitive. Levels: "
                   "exact, substring, strong_overlap (generic wording plus the "
                   "distinctive tokens %s and an anomaly term), weak_overlap "
                   "(generic wording only -- recorded, not counted as a match)."
                   % "/".join(DISTINCTIVE)),
        "n_queries": len(queries),
        "n_answered": sum(1 for q in queries if q["answered"]),
        "indexes": sorted(by_index),
        "nearest_indexed_records": nearest,
        "verdict": verdict,
        "queries": queries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1, sort_keys=False) + "\n",
                   encoding="utf-8")
    print("wrote %s (%d queries, %d answered)"
          % (OUT.relative_to(C.ROOT), len(queries), artifact["n_answered"]))
    for q in queries:
        print("  %-15s %-46s status=%-5s hits=%-5s target_matches=%s"
              % (q["index"], q["label"][:46], q["http_status"],
                 q["n_hits"], q.get("n_target_matches")))
    print("verdict: no_indexed_record=%s" % verdict["no_indexed_record"])
    if nearest.get("answered"):
        print("nearest indexed Ugandan weather records: scanned %d of %s, "
              "target-authored: %d"
              % (nearest["n_scanned"], nearest["index_count"],
                 sum(1 for r in nearest["records"] if r["author_matches_target"])))
        for r in nearest["records"][:3]:
            print("   - %s (%s, %s) %s" % (r["title"][:70], r["year"],
                                           r["type"], r["authors"][:60]))
    if unanswered:
        raise SystemExit("REFUSED: required index(es) did not answer: %s. "
                         "Do not claim the study is unindexed on this run."
                         % ", ".join(sorted(set(unanswered))))
    print("REF_CHECK_OK")


if __name__ == "__main__":
    main()
