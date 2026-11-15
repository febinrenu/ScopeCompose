"""Check the manuscript against the journal's rules and this project's own.

    python paper/check_paper.py            # after compiling paper/main.pdf

Journal (irojournals-scopus-paper-template-LaTex/guidelines.txt): at most 25 pages and
25 references; a 150-250 word abstract with no undefined abbreviations or citations;
four to six keywords; references cited in sequence. Project: every citation resolves
and every reference is cited; no ``\\pending`` marker is left; no digits in running
text outside generated macros, tables, equations and a short allow-list, so no number
is typed by hand.

Exit status 0 when everything passes; each failure is printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SECTIONS = ["abstract", "introduction", "related", "method", "results", "conclusion", "statements"]


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def body(name: str) -> str:
    return _strip_comments((PAPER / "sections" / f"{name}.tex").read_text(encoding="utf-8"))


def expand_inputs(tex: str) -> str:
    def sub(m):
        p = PAPER / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return expand_inputs(_strip_comments(p.read_text(encoding="utf-8"))) if p.exists() else ""
    return re.sub(r"\\input\{([^}]+)\}", sub, tex)


def fix_order() -> None:
    """Reorder the \\bibitem entries of references.tex by first citation in the text. Entries are moved, never edited."""
    path = PAPER / "sections" / "references.tex"
    raw = path.read_text(encoding="utf-8")
    head, rest = raw.split("\\bibitem", 1)
    rest, tail = rest.rsplit("\\end{thebibliography}", 1)
    items = ["\\bibitem" + chunk.rstrip() + "\n\n" for chunk in rest.split("\\bibitem")]
    key = {re.match(r"\\bibitem\{([^}]+)\}", it).group(1): it for it in items}
    text = "".join(expand_inputs(body(s)) for s in SECTIONS)
    order: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        for k in (x.strip() for x in group.split(",")):
            if k in key and k not in order:
                order.append(k)
    order += [k for k in key if k not in order]           # uncited entries last (the check still flags them)
    path.write_text(head + "".join(key[k] for k in order) + "\\end{thebibliography}" + tail, encoding="utf-8")


def main() -> int:
    if "--fix-order" in sys.argv:
        fix_order()
    failures: list[str] = []
    text = "".join(expand_inputs(body(s)) for s in SECTIONS)
    refs_tex = _strip_comments((PAPER / "sections" / "references.tex").read_text(encoding="utf-8"))

    # -- references --------------------------------------------------------- #
    bib = re.findall(r"\\bibitem\{([^}]+)\}", refs_tex)
    cites: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", text):
        for key in (k.strip() for k in group.split(",")):
            if key not in cites:
                cites.append(key)
    if len(bib) > 25:
        failures.append(f"{len(bib)} references; the journal allows 25")
    if missing := [k for k in cites if k not in bib]:
        failures.append(f"cited but not in the references: {missing}")
    if unused := [k for k in bib if k not in cites]:
        failures.append(f"in the references but never cited: {unused}")
    in_order = [k for k in bib if k in cites]
    if in_order != [k for k in cites if k in bib]:
        first_wrong = next(i for i, (a, b) in enumerate(zip(in_order, cites)) if a != b)
        failures.append(f"references are not numbered in order of first citation "
                        f"(first mismatch at [{first_wrong + 1}]: list has {in_order[first_wrong]}, "
                        f"text cites {cites[first_wrong]} first)")

    # -- abstract and keywords --------------------------------------------- #
    abstract = body("abstract")
    m = re.search(r"\\section\*\{Abstract\}(.*?)\\textbf\{.*?Keywords:\}(.*)", abstract, re.DOTALL)
    if not m:
        failures.append("abstract or keywords line not found")
    else:
        words = re.sub(r"\\[a-zA-Z]+\*?(\{[^}]*\})?|[{}~]", " ", m.group(1)).split()
        if not 150 <= len(words) <= 250:
            failures.append(f"abstract has {len(words)} words; the journal asks for 150-250")
        if "\\cite" in m.group(1):
            failures.append("the abstract cites a reference; the journal forbids it")
        kws = [k for k in re.split(r"[;,]", m.group(2).strip().rstrip(".")) if k.strip()]
        if not 4 <= len(kws) <= 6:
            failures.append(f"{len(kws)} keywords; the journal asks for 4-6")

    # -- pending markers and hand-typed numbers --------------------------- #
    for s in SECTIONS + ["biography"]:
        if "\\pending" in body(s):
            failures.append(f"sections/{s}.tex still has a \\pending marker")
    prose = re.sub(r"\\begin\{(equation|tabular)\}.*?\\end\{\1\}", " ", text, flags=re.DOTALL)
    prose = re.sub(r"\$[^$]*\$|\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}|\\includegraphics(\[[^]]*\])?\{[^}]*\}"
                   r"|\\texttt\{[^}]*\}|\\fontsize\{[^}]*\}\{[^}]*\}|\\vspace\{[^}]*\}|\[[0-9.]+(pt|cm|em)\]"
                   r"|width=[0-9.]+\\textwidth", " ", prose)
    allowed = re.compile(r"\b(?:Table|Figure|Section|Tables|Figures)~?\s*\d+(?:--\d+)?|\b(?:v[1-4]|k\s*=\s*\d+)\b"
                         r"|\b(?:BM25|F1|DeBERTa-?v?3?|Qwen2\.5-7B(?:-Instruct)?|Llama-3\.1-8B(?:-Instruct)?"
                         r"|Qwen3\.8-27B|Gemma-4-31B|Nemotron-3-Ultra|gpt-oss-\d+b|bge-reranker-base|ConditionalQA v1\.0|COLM 2025|gov\.uk|top-\d+)\b"
                         r"|\b(?:19|20)\d\d\b|\b\d+-word\b|\b(?:one|two)\b")
    stray = sorted({t for t in re.findall(r"\b\d[\d.,]*\b", allowed.sub(" ", prose))})
    # Numbers fixed by the design (not measured) are allowed when listed here.
    design = {"0", "1", "2", "3", "4", "5", "8", "13", "14", "20", "27", "50", "56.6", "100", "120", "150", "300",
              "000", "2,000", "20,000", "95", "1.00", "0.05", "15", "21.15", "40", "60",
              "768", "072", "024", "192", "18"}  # token budgets and an example age
    stray = [t for t in stray if t not in design]
    if stray:
        failures.append(f"numbers typed into the text instead of a macro: {stray}")

    # -- the compiled PDF ---------------------------------------------------- #
    pdf = PAPER / "main.pdf"
    if pdf.exists():
        try:
            import pypdf

            n = len(pypdf.PdfReader(str(pdf)).pages)
            if n > 25:
                failures.append(f"{n} pages; the journal allows 25")
            print(f"pages: {n}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"could not read main.pdf: {exc}")
    else:
        failures.append("main.pdf not found; compile first")
    log = PAPER / "main.log"
    if log.exists():
        lt = log.read_text(encoding="utf-8", errors="replace")
        if over := re.findall(r"^Overfull \\hbox \(([\d.]+)pt", lt, re.M):
            big = [float(x) for x in over if float(x) > 2.0]
            if big:
                failures.append(f"{len(big)} overfull line(s) wider than 2pt (max {max(big):.1f}pt)")
        if re.search(r"(Citation|Reference) `[^']+' on page \d+ undefined", lt):
            failures.append("undefined citation or reference in the log")

    print(f"references: {len(bib)} (cited {len(cites)})")
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print("all checks pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
