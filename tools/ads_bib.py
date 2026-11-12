"""Generate ``docs/paper/referencias.bib`` from NASA ADS, with every entry verified.

Each reference of the paper has a BibTeX key (Author+year), what is expected of it (first
author, year) and its ADS bibcode or, if unknown, an ADS query that must return a single
result. The script resolves the queries, checks first author and year, exports the ADS BibTeX
with those keys and writes it. The token is read at run time (``ADS_DEV_KEY`` or
``~/.ads/dev_key`` / ``~/.ads/devkey.txt``) and is never printed or written.

    python tools/ads_bib.py            # writes docs/paper/referencias.bib and docs/paper/referencias_ads.json
    python tools/ads_bib.py --check    # only verifies, without writing
"""
import json
import os
import re
import sys
import unicodedata
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads_check import API, query, token  # noqa: E402

OUT_BIB = os.path.join("docs", "paper", "referencias.bib")
OUT_JSON = os.path.join("docs", "paper", "referencias_ads.json")

# key: (expected first author, year, bibcode or ADS query)
REFS = {
    "Aly1989": ("Aly", 1989, 'first_author:"Aly" year:1989 bibstem:SoPh volume:120 page:19'),
    "Aschwanden2013": ("Aschwanden", 2013, "2013ApJ...763..115A"),
    "Aschwanden2014": ("Aschwanden", 2014, 'first_author:"Aschwanden" year:2014 bibstem:ApJ volume:785 page:34'),
    "Astropy2022": ("Astropy", 2022, 'first_author:"Astropy Collaboration" year:2022 bibstem:ApJ volume:935 page:167'),
    "Bobra2014": ("Bobra", 2014, "2014SoPh..289.3549B"),
    "Chifu2017": ("Chifu", 2017, "2017ApJ...837...10C"),
    "DeRosa2009": ("DeRosa", 2009, "2009ApJ...696.1780D"),
    "DeRosa2015": ("DeRosa", 2015, "2015ApJ...811..107D"),
    "DeVore2000": ("DeVore", 2000, 'first_author:"DeVore" year:2000 bibstem:ApJ volume:539 page:944'),
    "Finn1985": ("Finn", 1985, 'first_author:"Finn" author:"Antonsen" year:1985 title:"helicity"'),
    "Fleishman2017": ("Fleishman", 2017, "2017ApJ...839...30F"),
    "Gilchrist2020": ("Gilchrist", 2020, "2020ApJ...900..136G"),
    "Harris2020": ("Harris", 2020, 'first_author:"Harris" year:2020 bibstem:Natur volume:585 page:357'),
    "Hoeksema2014": ("Hoeksema", 2014, 'first_author:"Hoeksema" year:2014 bibstem:SoPh volume:289 page:3483'),
    "Hunter2007": ("Hunter", 2007, 'first_author:"Hunter" year:2007 bibstem:CSE volume:9 page:90'),
    "Hurlburt2012": ("Hurlburt", 2012, 'first_author:"Hurlburt" year:2012 bibstem:SoPh volume:275 page:67'),
    "Iyer2026": ("Iyer", 2026, "2026ApJ..1004..112I"),
    "Lemen2012": ("Lemen", 2012, "2012SoPh..275...17L"),
    "Liu1989": ("Liu", 1989, "1989MatPr..45..503L"),
    "Low1990": ("Low", 1990, "1990ApJ...352..343L"),
    "MacTaggart2025": ("MacTaggart", 2025, "2025SoPh..300...48M"),
    "Malanushenko2012": ("Malanushenko", 2012, "2012ApJ...756..153M"),
    "Moraitis2019": ("Moraitis", 2019, "2019A&A...624A..51M"),
    "Moraitis2021": ("Moraitis", 2021, "2021A&A...649A.107M"),
    "Paszke2019": ("Paszke", 2019, "2019arXiv191201703P"),
    "Pedregosa2011": ("Pedregosa", 2011, 'first_author:"Pedregosa" year:2011 bibstem:JMLR volume:12 page:2825'),
    "Pesnell2012": ("Pesnell", 2012, 'first_author:"Pesnell" year:2012 bibstem:SoPh volume:275 page:3'),
    "Sandman2009": ("Sandman", 2009, 'first_author:"Sandman" year:2009 bibstem:SoPh volume:259'),
    "Scherrer2012": ("Scherrer", 2012, 'first_author:"Scherrer" year:2012 bibstem:SoPh volume:275 page:207'),
    "Schrijver2006": ("Schrijver", 2006, "2006SoPh..235..161S"),
    "Schrijver2008": ("Schrijver", 2008, "2008ApJ...675.1637S"),
    "Sun2012": ("Sun", 2012, 'first_author:"Sun, X" year:2012 bibstem:ApJ volume:748 page:77'),
    "SunPy2020": ("SunPy", 2020, "2020ApJ...890...68S"),
    "Thalmann2019": ("Thalmann", 2019, "2019ApJ...880L...6T"),
    "Titov1999": ("Titov", 1999, "1999A&A...351..707T"),
    "Valori2012": ("Valori", 2012, 'first_author:"Valori" year:2012 bibstem:SoPh title:"helicity"'),
    "Valori2013": ("Valori", 2013, "2013A&A...553A..38V"),
    "Virtanen2020": ("Virtanen", 2020, "2020NaMet..17..261V"),
    "Warren2018": ("Warren", 2018, "2018ApJ...860...46W"),
    "Wheatland2000": ("Wheatland", 2000, "2000ApJ...540.1150W"),
    "Wiegelmann2004": ("Wiegelmann", 2004, "2004SoPh..219...87W"),
    "Wiegelmann2006": ("Wiegelmann", 2006, "2006SoPh..233..215W"),
    "Wiegelmann2010": ("Wiegelmann", 2010, "2010A&A...516A.107W"),
    "Wiegelmann2012": ("Wiegelmann", 2012, "2012LRSP....9....5W"),
    "Zhu2018": ("Zhu", 2018, "2018ApJ...866..130Z"),
}
FL = "bibcode,title,first_author,author,year,pub,volume,page,doi"
# Unified author spellings (ADS writes "De Rosa" in 2009ApJ...696.1780D; the author signs "DeRosa")
AUTHOR_FIXES = [("{De Rosa}, Marc L.", "{DeRosa}, Marc L.")]


def plain(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c) and c != " ").lower()


def resolve():
    out, problems = {}, []
    for key, (author, year, ref) in REFS.items():
        docs = query(f'bibcode:"{ref}"' if re.match(r"^\d{4}[A-Za-z&.]", ref) else ref, fl=FL, rows=5)
        if len(docs) != 1:
            problems.append(f"{key}: {len(docs)} resultados para {ref!r}: " + "; ".join(d["bibcode"] for d in docs))
            if not docs:
                continue
        d = docs[0]
        first = plain(d.get("first_author", ""))
        if plain(author) not in first or int(d["year"]) != year:
            problems.append(f"{key}: esperado {author} {year}, ADS da {d.get('first_author')} {d.get('year')} ({d['bibcode']})")
        out[key] = {"bibcode": d["bibcode"], "first_author": d.get("first_author"), "year": d.get("year"),
                    "title": (d.get("title") or [""])[0], "pub": d.get("pub"), "volume": d.get("volume"),
                    "page": (d.get("page") or [""])[0], "doi": (d.get("doi") or [""])[0]}
        print(f"  {key:18s} {d['bibcode']:20s} {d.get('first_author', ''):22s} {d.get('pub', '')[:28]:28s} {out[key]['title'][:60]}")
    return out, problems


def export_bibtex(entries):
    """ADS BibTeX with our keys (one request per entry to set the key)."""
    chunks = []
    for key, e in entries.items():
        body = json.dumps({"bibcode": [e["bibcode"]]}).encode()
        req = urllib.request.Request(f"{API}/export/bibtex", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            bib = json.load(r)["export"]
        bib = re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", bib.strip(), count=1, flags=re.M)
        for a, b in AUTHOR_FIXES:
            bib = bib.replace(a, b)
        chunks.append(bib)
    return "\n\n".join(chunks) + "\n"


if __name__ == "__main__":
    entries, problems = resolve()
    for p in problems:
        print("AVISO", p)
    if "--check" in sys.argv:
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit("No se escribe el .bib: resuelve los avisos")
    bib = export_bibtex(entries)
    header = ("% Generado por tools/ads_bib.py desde NASA ADS; no editar a mano (regenerar).\n"
              f"% {len(entries)} entradas verificadas (primer autor y año).\n\n")
    open(OUT_BIB, "w", encoding="utf-8").write(header + bib)
    json.dump(entries, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("escrito", OUT_BIB, "y", OUT_JSON, f"({len(entries)} entradas)")
