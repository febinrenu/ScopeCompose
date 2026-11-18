"""Build a Word version of the manuscript from the same LaTeX sources as main.pdf.

    python paper/make_docx.py        # writes paper/submission/<name>.docx

The sources are flattened, the generated number macros expanded, citations, section and equation numbers
resolved as LaTeX numbers them, figures rasterised, and the result converted by pandoc (pypandoc_binary).
The page, fonts and spacing follow the journal template: A4, Times New Roman 12 pt, 1.5 line spacing.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pymupdf
import pypandoc
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "submission" / "Conditional_Answer_Fidelity_in_RAG.docx"
FONT = "Times New Roman"


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def read(rel: str) -> str:
    p = PAPER / (rel if rel.endswith(".tex") else rel + ".tex")
    tex = strip_comments(p.read_text(encoding="utf-8"))
    return re.sub(r"\\input\{([^}]+)\}", lambda m: read(m.group(1)), tex)


def braced(s: str, i: int) -> tuple[str, int]:
    """Content of the brace group opening at s[i] and the index after it."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def number_macros() -> dict[str, str]:
    src = strip_comments((PAPER / "numbers.tex").read_text(encoding="utf-8"))
    macros, i = {}, 0
    while (i := src.find("\\newcommand{", i)) >= 0:
        name, j = braced(src, i + len("\\newcommand"))
        macros[name.lstrip("\\")], i = braced(src, j)
    return macros


def unwrap(tex: str, cmd: str) -> str:
    """Replace \\cmd{x} by x, for nested groups."""
    while (i := tex.find(cmd + "{")) >= 0:
        inner, j = braced(tex, i + len(cmd))
        tex = tex[:i] + inner + tex[j:]
    return tex


def table(block: str) -> str:
    caption, rest = block.split("\\\\", 1)
    caption = caption.replace("\\begin{table}[!htbp]\\centering", "").strip()
    body = rest[rest.index("\\begin{tabular}"):rest.rindex("\\end{tabular}") + len("\\end{tabular}")]
    spec, k = braced(body, len("\\begin{tabular}"))
    cols = re.findall(r"[LPM]\{[^}]*\}|[lcr]|\*\{\d+\}\{[^}]*\}", spec)
    n = sum(int(re.match(r"\*\{(\d+)\}", c).group(1)) if c.startswith("*") else 1 for c in cols)
    body = "\\begin{tabular}{l" + "c" * (n - 1) + "}" + body[k:]
    return f"\n\n{caption}\n\n{body}\n\n"


def figure(block: str, tmp: Path) -> str:
    name = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", block).group(1)
    width = float(re.search(r"width=([\d.]+)\\textwidth", block).group(1))
    png = tmp / (Path(name).stem + ".png")
    page = pymupdf.open(PAPER / "figures" / name)[0]
    page.get_pixmap(dpi=300).save(png)
    caption = block[block.index("\\textbf{Figure"):block.rindex("\\end{figure}")].strip()
    return (f"\n\n\\begin{{center}}\\includegraphics[width={width * 17:.1f}cm]{{{png.as_posix()}}}\\end{{center}}"
            f"\n\n{caption}\n\n")


def biographies(tex: str) -> str:
    rows = []
    for m in re.finditer(r"\\bio\{(author\d)\}", tex):
        text, _ = braced(tex, m.end())
        img = (PAPER / "figures" / "authors" / f"{m.group(1)}.jpg").as_posix()
        rows.append(f"\\includegraphics[width=2.3cm]{{{img}}} & {text} \\\\")
    return ("\\section*{Author's biography}\n\n\\begin{tabular}{p{2.6cm}p{13.4cm}}\n"
            + "\n".join(rows) + "\n\\end{tabular}\n")


def build_tex(tmp: Path) -> str:
    tex = "\n\n".join(read(f"sections/{s}") for s in
                      ("abstract", "introduction", "related", "method", "results", "conclusion", "statements"))
    refs = read("sections/references")
    bio = read("sections/biography")

    # generated numbers
    macros = number_macros()
    pat = re.compile(r"\\(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")(?![a-zA-Z])(\{\})?")
    for _ in range(2):  # a macro's value may use another macro
        tex = pat.sub(lambda m: macros[m.group(1)], tex)

    # citations, numbered by order in the reference list (already order of first citation)
    keys = re.findall(r"\\bibitem\{([^}]+)\}", refs)
    tex = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + ", ".join(str(keys.index(k.strip()) + 1)
                                                                  for k in m.group(1).split(",")) + "]", tex)

    # section numbers and labels
    sec = sub = 0
    labels: dict[str, str] = {}
    out = []
    for line in tex.split("\n"):
        if m := re.match(r"\\section\{(.*)\}\s*$", line):
            sec, sub = sec + 1, 0
            num, title = f"{sec}.", unwrap(m.group(1), "")
            line = f"\\section*{{{num} {title.strip('{}')}}}"
            cur = str(sec)
        elif m := re.match(r"\\subsection\{(.*)\}\s*$", line):
            sub += 1
            cur = f"{sec}.{sub}"
            line = f"\\subsection*{{{cur} {m.group(1)}}}"
        elif m := re.match(r"\\label\{([^}]+)\}\s*$", line):
            labels[m.group(1)] = cur
            continue
        out.append(line)
    tex = "\n".join(out)
    tex = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], tex)

    # numbered equations
    eq = 0

    def equation(m):
        nonlocal eq
        eq += 1
        return f"\n\\[ {m.group(1).strip().rstrip(',.')} \\qquad ({eq}) \\]\n"
    tex = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", equation, tex, flags=re.S)

    tex = re.sub(r"\\begin\{table\}.*?\\end\{table\}", lambda m: table(m.group(0)), tex, flags=re.S)
    tex = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", lambda m: figure(m.group(0), tmp), tex, flags=re.S)

    # reference list as numbered paragraphs
    items = re.split(r"\\bibitem\{[^}]+\}", refs.split("\\end{thebibliography}")[0])[1:]
    tex += "\n\n\\section*{References}\n\n" + "\n\n".join(f"[{i}] {t.strip()}" for i, t in enumerate(items, 1))
    tex += "\n\n" + biographies(bio)

    # layout commands with no meaning in Word
    tex = re.sub(r"\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont", "", tex)
    tex = re.sub(r"\\(vspace|setlength)\{[^}]*\}(\{[^}]*\})?", "", tex)
    tex = re.sub(r"\\(par|noindent|centering|hfill)(?![a-zA-Z])", "", tex)
    tex = re.sub(r"\\section\*\{\s*\\textbf\{(.*?)\}\s*\}", r"\\section*{\1}", tex)
    tex = re.sub(r"\\begin\{enumerate\}\[(.*?)\](.*?)\\end\{enumerate\}", enumerate_labels, tex, flags=re.S)
    return tex


ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"]


def enumerate_labels(m) -> str:
    """Lists as paragraphs with their LaTeX labels, (1) or (i), which Word then cannot renumber."""
    items = [t.strip() for t in m.group(2).split("\\item")[1:]]
    roman = "i" in m.group(1)
    return "\n\n" + "\n\n".join(f"({ROMAN[k] if roman else k + 1})~{t}" for k, t in enumerate(items)) + "\n\n"


def front_matter() -> str:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"\\fontsize\{24\}\{24\}\\selectfont\\textbf\s*\{(.*?)\}\s*\\end\{center\}", main, re.S).group(1).strip()
    title = title.replace("--", "\u2013")
    block = main[main.index("\\else", main.index("\\ifdefined\\reviewcopy\n\\begin{center}")):]
    block = block[:block.index("\\fi")]
    names = re.search(r"\\textbf\{%?\s*(Febin.*?)\}\s*\\vspace", block, re.S).group(1).strip()
    affil = re.search(r"\\selectfont \{\s*(.*?)\}\\vspace", block, re.S).group(1)
    emails = re.search(r"\\textbf\{\{E-mail:\}\}(.*?)\\\\\s*\n", block, re.S).group(1).strip()
    corr = re.search(r"(\$\^\*\$Corresponding author:[^}]*)\}", block).group(1).strip()
    affil = [a.strip() for a in affil.split("\\\\") if a.strip()]
    return "\n\n".join([title, names.replace("~", " "), *affil, "E-mail: " + emails, corr]) + "\n\n"


def style(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.54)
    sec.top_margin = sec.bottom_margin = Cm(2.65)
    # Times everywhere: explicit fonts and no theme fonts, which would otherwise win in headings
    for rf in doc.styles.element.iter(qn("w:rFonts")):
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            rf.attrib.pop(qn(a), None)
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), FONT)
    for s in doc.styles:
        if s.type == 2 and s.name == "Verbatim Char":
            s.font.name = "Courier New"
            s.font.size = Pt(10.5)
        elif s.type == 1:
            s.font.name = FONT
            s.font.color.rgb = RGBColor(0, 0, 0)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Bibliography"):
        if name in [s.name for s in doc.styles]:
            s = doc.styles[name]
            s.font.size = Pt(12)
            pf = s.paragraph_format
            # LaTeX's onehalfspace at 12 pt is an 18 pt baseline; Word's "1.5 lines" would be about 21 pt
            pf.line_spacing = 1.3
            pf.space_before, pf.space_after = Pt(0), Pt(10)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        s = doc.styles[name]
        s.font.size, s.font.bold, s.font.italic = Pt(12), True, False
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 10)
        s.paragraph_format.space_after = Pt(6)
        s.paragraph_format.keep_with_next = True
    if "Footnote Text" in [s.name for s in doc.styles]:
        doc.styles["Footnote Text"].font.size = Pt(10)


def postprocess(path: Path) -> None:
    doc = Document(path)
    style(doc)
    paras = doc.paragraphs
    # title block: title, names, affiliations, e-mail, corresponding author
    sizes = [24, 16, 11, 11, 11, 9, 9]
    for p, size in zip(paras, sizes):
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(4 if size in (11, 9) else 14)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if size == 24 else WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.size, r.font.name = Pt(size), FONT
            r.font.bold = size in (24, 16)
    for p in paras:
        t = p.text
        if re.match(r"(Table|Figure) \d+\.", t):
            p.paragraph_format.keep_with_next = t.startswith("Table")
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if re.match(r"\[\d+\] ", t):
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.9), Cm(-0.9)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.0  # LaTeX sets the list at 11 pt on a 13 pt baseline
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(11)
        if re.match(r"\((\d|[iv]+)\)\s", t):
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.9), Cm(-0.9)
            p.paragraph_format.space_after = Pt(6)
        if t.startswith("Keywords:"):
            p.runs[0].font.bold = True
    # centred page number in the footer, as in the template
    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): tag})
        else:
            el = run._r.makeelement(qn("w:instrText"), {qn("xml:space"): "preserve"})
            el.text = text
        run._r.append(el)
    run.font.name, run.font.size = FONT, Pt(11)
    for ti, tbl in enumerate(doc.tables):
        bio = ti == len(doc.tables) - 1
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for row in tbl.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0 if not bio else 4)
                    p.paragraph_format.line_spacing = 1.0 if not bio else 1.15
                    if not bio:
                        num = p.text[:1].isdigit() or p.text[:1] in "+−-" or p.text in ("--",)
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if num else WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.size = Pt(9 if not bio else 12)
        if bio:  # no borders on the biography layout table
            tblPr = tbl._tbl.tblPr
            for b in tblPr.findall(qn("w:tblBorders")):
                tblPr.remove(b)
            tbl.style = None
            for row in tbl.rows:  # a biography is never split across pages
                trPr = row._tr.get_or_add_trPr()
                trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))
            continue
        # a table never breaks across pages, and has booktabs-style rules
        rows = tbl.rows
        header = sum(1 for r in rows[:3] if not any(c.text[:1].isdigit() for c in r.cells[1:]))
        for ri, row in enumerate(rows):
            trPr = row._tr.get_or_add_trPr()
            trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = ri < len(rows) - 1
                edges = {}
                if ri == 0:
                    edges["top"] = 8
                if ri == max(header, 1) - 1:
                    edges["bottom"] = 4
                if ri == len(rows) - 1:
                    edges["bottom"] = 8
                if edges:
                    tcPr = cell._tc.get_or_add_tcPr()
                    borders = tcPr.makeelement(qn("w:tcBorders"), {})
                    for side, sz in edges.items():
                        borders.append(borders.makeelement(qn(f"w:{side}"), {qn("w:val"): "single", qn("w:sz"): str(sz),
                                                                             qn("w:space"): "0", qn("w:color"): "000000"}))
                    tcPr.append(borders)
    for tbl in doc.tables:  # space between a table and the text after it
        nxt = tbl._tbl.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            pPr = nxt.get_or_add_pPr()
            for old in pPr.findall(qn("w:spacing")):
                pPr.remove(old)
            pPr.append(pPr.makeelement(qn("w:spacing"), {qn("w:before"): "200"}))
    for p in doc.paragraphs:  # figures centred
        if p._p.findall(".//" + qn("w:drawing")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
    doc.save(path)


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tex = "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n" \
              + front_matter() + build_tex(tmp) + "\n\\end{document}\n"
        (tmp / "flat.tex").write_text(tex, encoding="utf-8")
        pypandoc.convert_file(str(tmp / "flat.tex"), "docx", outputfile=str(OUT), format="latex",
                              extra_args=[f"--resource-path={tmp}", "--wrap=none"])
    postprocess(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
