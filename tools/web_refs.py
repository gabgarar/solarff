"""References of the documentation site, verified against NASA ADS.

Writes ``docs/web/references.json`` (read by ``tools/mkdocs_hooks.py`` to turn every citation of the
site into a link) and ``docs/web/references.md`` (the page the links point to). The references of
the paper come from ``tools/ads_bib.py``; the ones cited only in the documentation are listed here.
Every ADS entry is checked (first author and year) before writing. The token is read at run time
and is never printed or written.

    python tools/web_refs.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ads_bib import REFS, plain  # noqa: E402
from ads_check import query  # noqa: E402

OUT_JSON = os.path.join("docs", "web", "references.json")
OUT_MD = os.path.join("docs", "web", "references.md")

# Cited in the documentation or in the docstrings but not in the paper: key -> (first author, year, bibcode)
WEB_REFS = {
    "Alissandrakis1981": ("Alissandrakis", 1981, "1981A&A...100..197A"),
    "Aly1989": ("Aly", 1989, "1989SoPh..120...19A"),
    "Amari2006": ("Amari", 2006, "2006A&A...446..691A"),
    "Berger1984": ("Berger", 1984, "1984JFM...147..133B"),
    "Berger2006": ("Berger", 2006, "2006JPhA...39.8321B"),
    "DeVore2000": ("DeVore", 2000, "2000ApJ...539..944D"),
    "Gold1960": ("Gold", 1960, "1960MNRAS.120...89G"),
    "Hagino2004": ("Hagino", 2004, "2004PASJ...56..831H"),
    "Kliem2006": ("Kliem", 2006, "2006PhRvL..96y5002K"),
    "Liu2016": ("Liu", 2016, "2016ApJ...818..148L"),
    "Nakagawa1972": ("Nakagawa", 1972, "1972SoPh...25..127N"),
    "Pariat2012": ("Pariat", 2012, "2012A&A...541A..78P"),
    "Pariat2017": ("Pariat", 2017, "2017A&A...601A.125P"),
    "Valori2010": ("Valori", 2010, "2010A&A...519A..44V"),
    "Guo2017": ("Guo", 2017, "2017arXiv170604599G"),
    "Titov2002": ("Titov", 2002, "2002JGRA..107.1164T"),
    "Wheatland2007": ("Wheatland", 2007, "2007SoPh..245..251W"),
    "Wheatland2009": ("Wheatland", 2009, "2009ApJ...700L..88W"),
}
# Not in ADS: written by hand
MANUAL = {
    "ElYaniv2010": {"authors": ["El-Yaniv, R.", "Wiener, Y."], "year": 2010,
                    "title": "On the Foundations of Noise-free Selective Classification",
                    "pub": "Journal of Machine Learning Research", "volume": "11", "page": "1605"},
    "Geifman2017": {"authors": ["Geifman, Y.", "El-Yaniv, R."], "year": 2017,
                    "title": "Selective Classification for Deep Neural Networks",
                    "pub": "Advances in Neural Information Processing Systems", "volume": "30", "page": ""},
    "NiculescuMizil2005": {"authors": ["Niculescu-Mizil, A.", "Caruana, R."], "year": 2005,
                           "title": "Predicting Good Probabilities with Supervised Learning",
                           "pub": "Proceedings of the 22nd International Conference on Machine Learning", "volume": "", "page": "625"},
    "Grad1958": {"authors": ["Grad, H.", "Rubin, H."], "year": 1958,
                 "title": "Hydromagnetic Equilibria and Force-Free Fields",
                 "pub": "Proc. 2nd UN International Conference on the Peaceful Uses of Atomic Energy", "volume": "31",
                 "page": "190"},
}
FL = "bibcode,title,first_author,author,year,pub,volume,page,doi"
JOURNALS = {"The Astrophysical Journal": "ApJ", "Astronomy and Astrophysics": "A&A", "Solar Physics": "Sol. Phys.",
            "Monthly Notices of the Royal Astronomical Society": "MNRAS", "Physical Review Letters": "Phys. Rev. Lett.",
            "Publications of the Astronomical Society of Japan": "PASJ", "Journal of Fluid Mechanics": "J. Fluid Mech.",
            "Journal of Geophysical Research (Space Physics)": "J. Geophys. Res.", "Living Reviews in Solar Physics": "Living Rev. Sol. Phys.",
            "Journal of Physics A Mathematical General": "J. Phys. A", "Mathematical Programming": "Math. Program.",
            "arXiv e-prints": "arXiv"}


def initials(name):
    """'Wiegelmann, Thomas' -> 'Wiegelmann, T.'"""
    if "," not in name:
        return name
    last, first = [x.strip() for x in name.split(",", 1)]
    ini = " ".join(p[0] + "." for p in re.split(r"[\s\-]+", first) if p)
    return f"{last}, {ini}" if ini else last


def resolve():
    out, problems = {}, []
    for key, (author, year, ref) in {**REFS, **WEB_REFS}.items():
        docs = query(f'bibcode:"{ref}"' if re.match(r"^\d{4}[A-Za-z&.]", ref) else ref, fl=FL, rows=5)
        if len(docs) != 1:
            problems.append(f"{key}: {len(docs)} results for {ref!r}")
            continue
        d = docs[0]
        if plain(author) not in plain(d.get("first_author", "")) or int(d["year"]) != year:
            problems.append(f"{key}: expected {author} {year}, ADS gives {d.get('first_author')} {d.get('year')} ({d['bibcode']})")
            continue
        out[key] = {"authors": [initials(a).replace("De Rosa,", "DeRosa,") for a in d.get("author", [])], "year": int(d["year"]),
                    "title": re.sub(r"<[^>]+>", "", (d.get("title") or [""])[0]), "pub": d.get("pub", ""),
                    "volume": d.get("volume", ""), "page": (d.get("page") or [""])[0], "doi": (d.get("doi") or [""])[0],
                    "bibcode": d["bibcode"]}
        print(f"  {key:18s} {d['bibcode']:20s} {d.get('first_author', '')[:22]:22s} {out[key]['title'][:60]}")
    return out, problems


def author_text(a):
    if len(a) == 1:
        return a[0]
    if len(a) <= 3:
        return ", ".join(a[:-1]) + " & " + a[-1]
    return a[0] + " et al."


def page_md(refs):
    lines = ["# References", "",
             "Every citation in this documentation links to its entry here. Entries with an ADS link were "
             "checked against NASA ADS (first author and year).", ""]
    for key in sorted(refs, key=lambda k: (plain(refs[k]["authors"][0]), refs[k]["year"])):
        r = refs[key]
        pub = JOURNALS.get(r["pub"], r["pub"])
        where = ", ".join(x for x in (pub, str(r.get("volume") or ""), str(r.get("page") or "")) if x)
        links = []
        if r.get("doi"):
            links.append(f"[DOI](https://doi.org/{r['doi']})")
        if r.get("bibcode"):
            links.append(f"[ADS](https://ui.adsabs.harvard.edu/abs/{r['bibcode'].replace('&', '%26')})")
        tail = (" " + " · ".join(links)) if links else ""
        lines.append(f'- <a id="{key.lower()}"></a>{author_text(r["authors"])} ({r["year"]}). *{r["title"]}*. {where}.{tail}')
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    refs, problems = resolve()
    for p in problems:
        print("WARNING", p)
    if problems:
        sys.exit("Nothing written: resolve the warnings")
    refs.update(MANUAL)
    json.dump(refs, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, sort_keys=True)
    open(OUT_MD, "w", encoding="utf-8").write(page_md(refs))
    print("written", OUT_JSON, "and", OUT_MD, f"({len(refs)} references)")
