"""MkDocs hook: every citation of the site ("Wiegelmann (2004)", "Valori et al. 2013", "Low & Lou
(1990)"...) becomes a link to its entry in the references page.

The references are read from ``docs/web/references.json`` (written by ``tools/web_refs.py``). A
citation is recognised by the surname of the first author followed by the year; text inside links,
code, headings and scripts is left alone, and so are citations that are not in the list.
"""
import json
import os
import re

from mkdocs.utils import get_relative_url

REFS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "docs", "web", "references.json"), encoding="utf-8"))
TARGET = "references/"
SKIP = {"a", "code", "pre", "script", "style", "h1", "h2", "h3", "h4", "h5", "h6"}


def _surname(author):
    return author.split(",")[0].split(" Community")[0].strip()


KEYS = {(_surname(r["authors"][0]), str(r["year"])): k.lower() for k, r in REFS.items()}
_SUR = "|".join(sorted({re.escape(s) for s, _ in KEYS}, key=len, reverse=True))
# first-author surname, optional co-authors ("X, Y & Z", "X and Y", "X et al."), then "(year)" or "year"
CITE = re.compile(
    rf"(?<![\w\-])({_SUR})"
    r"((?:(?:,\s+|\s+&amp;\s+|\s+&\s+|\s+and\s+)[A-ZÀ-Ý][^\s,;()&<]*)*(?:\s+et\s+al\.)?)"
    r"(?:,?\s+\((\d{4})[a-z]?\)|,?\s+(\d{4})[a-z]?)(?!\d)")
TAG = re.compile(r"(<[^>]+>)")
NAME = re.compile(r"</?\s*([a-zA-Z0-9]+)")


def _link(text, base):
    def sub(m):
        key = KEYS.get((m.group(1), m.group(3) or m.group(4)))
        return f'<a class="citation" href="{base}#{key}">{m.group(0)}</a>' if key else m.group(0)
    return CITE.sub(sub, text)


def on_page_content(html, page, config, files):
    if page.file.src_uri == "references.md":
        return html
    base = get_relative_url(TARGET, page.url)
    out, depth = [], 0
    for part in TAG.split(html):
        if part.startswith("<"):
            m = NAME.match(part)
            if m and m.group(1).lower() in SKIP and not part.endswith("/>"):
                depth += -1 if part.startswith("</") else 1
            out.append(part)
        else:
            out.append(part if depth > 0 else _link(part, base))
    return "".join(out)
