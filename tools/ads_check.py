"""Check the references of the report against NASA ADS and search for related work.

The token is read at run time from ``ADS_DEV_KEY`` or from ``~/.ads/dev_key``
(or ``~/.ads/devkey.txt``); it is never written in the code or in the output.

    python tools/ads_check.py check            # verifies author/year/volume/page
    python tools/ads_check.py search "ADS query" [--rows 20]
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
API = "https://api.adsabs.harvard.edu/v1"


def token() -> str:
    if os.environ.get("ADS_DEV_KEY"):
        return os.environ["ADS_DEV_KEY"].strip()
    for name in ("dev_key", "devkey.txt"):
        p = os.path.expanduser(os.path.join("~", ".ads", name))
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
    raise SystemExit("No hay token de ADS (ADS_DEV_KEY o ~/.ads/dev_key)")


def query(q: str, fl="bibcode,title,author,year,pub,volume,page,doi,citation_count", rows=5, sort=None):
    params = {"q": q, "fl": fl, "rows": rows}
    if sort:
        params["sort"] = sort
    req = urllib.request.Request(f"{API}/search/query?{urllib.parse.urlencode(params)}",
                                 headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["response"]["docs"]


def _parse(ref: str):
    """'Surname, X., ... YYYY, Journal, vol, page' -> (surname, year, vol, page)."""
    surname = re.sub(r"\\[\"'`^~]|[{}\\]", "", ref.split(",")[0]).strip()
    m = re.search(r"\b(1[89]\d\d|20\d\d)[a-z]?\b", ref)
    year = m.group(1) if m else None
    tail = [t.strip() for t in ref[m.end():].split(",")] if m else []
    nums = [t for t in tail if re.fullmatch(r"(PMLR )?[A-Z]?\d+[A-Z]?", t)]
    vol = nums[0] if len(nums) >= 2 else None
    page = nums[-1] if nums else None
    return surname, year, vol, page


def check():
    from solarff.report.texts import REFERENCES
    bad = 0
    for key, ref in REFERENCES:
        s, y, v, p = _parse(ref)
        docs = query(f'first_author:"{s}" year:{y}' + (f' page:"{p}"' if p else ""))
        hit = None
        for d in docs:
            pages = d.get("page") or []
            if (not v or d.get("volume") == v) and (not p or p in pages):
                hit = d
                break
        if hit:
            print(f"OK   {key:22s} {hit['bibcode']}  doi={(hit.get('doi') or ['-'])[0]}")
        else:
            bad += 1
            cand = "; ".join(f"{d['bibcode']} v{d.get('volume')} p{(d.get('page') or ['?'])[0]}" for d in docs[:3])
            print(f"??   {key:22s} {s} {y} v{v} p{p} -> {cand or 'sin resultados'}")
        time.sleep(0.2)
    print(f"\n{len(REFERENCES) - bad}/{len(REFERENCES)} verificadas en ADS")


def search(q: str, rows: int):
    for d in query(q, rows=rows, sort="citation_count desc"):
        au = d.get("author") or ["?"]
        print(f"{d['bibcode']}  [{d.get('citation_count', 0)} cit]  {au[0]}{' et al.' if len(au) > 1 else ''}"
              f" {d.get('year')}: {(d.get('title') or [''])[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        rows = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 20
        search(sys.argv[2], rows)
    else:
        check()
