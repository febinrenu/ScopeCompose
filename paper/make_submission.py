"""Build the LaTeX source package the journal asks for, and test that it compiles on its own.

    python paper/make_submission.py      # writes paper/submission/<name>_LaTeX_source.zip

The package holds exactly what main.tex needs: the preamble, the generated numbers, every section, the
tables and figures it inputs, the author photos, the compiled main.pdf and a short README. It is unpacked
into an empty folder and compiled there with Tectonic when Tectonic is installed, so a missing file is
an error here rather than at the publisher.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "submission" / "Where_Do_the_Conditions_Go_LaTeX_source.zip"
README = """Where Do the Conditions Go? Stage-wise Attribution of Condition Loss in Chunked Retrieval-Augmented Generation
LaTeX source for submission to the Journal of Trends in Computer Science and Smart Technology.

Main file: main.tex. Compile with pdfLaTeX or XeLaTeX (run twice for references):
    pdflatex main.tex
main.pdf is the compiled manuscript.

Contents
  main.tex          preamble and title block (from the IRO Journals template)
  numbers.tex       numeric macros used in the text
  sections/         abstract, sections, statements, references, author biographies
  tables/           tables
  figures/          figures (PDF) and author photos (figures/authors)
"""


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def needed() -> list[Path]:
    """main.tex and everything it inputs or includes, recursively."""
    files, todo = [PAPER / "main.tex"], [PAPER / "main.tex"]
    while todo:
        tex = strip_comments(todo.pop().read_text(encoding="utf-8"))
        for name in re.findall(r"\\input\{([^}]+)\}", tex):
            p = PAPER / (name if name.endswith(".tex") else name + ".tex")
            if p not in files:
                files.append(p)
                todo.append(p)
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}#]+)\}", tex):
            files.append(PAPER / "figures" / name)
        if "\\bio{" in tex:  # biographies name their photos through a macro
            files += sorted((PAPER / "figures" / "authors").glob("author*.jpg"))
    return list(dict.fromkeys(files))


def main() -> int:
    files = needed() + [PAPER / "main.pdf"]
    missing = [f for f in files if not f.exists()]
    if missing:
        print("missing:", *missing, sep="\n  ")
        return 1
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(PAPER).as_posix())
        z.writestr("README.txt", README)
    print(f"{OUT} ({len(files) + 1} files)")

    tectonic = shutil.which("tectonic") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "tectonic" / "tectonic.exe")
    if not Path(tectonic).exists() and not shutil.which("tectonic"):
        print("Tectonic not found: package written but not test-compiled")
        return 0
    with tempfile.TemporaryDirectory() as d:
        zipfile.ZipFile(OUT).extractall(d)
        (Path(d) / "main.pdf").unlink()
        r = subprocess.run([tectonic, "-X", "compile", "main.tex"], cwd=d, capture_output=True, text=True)
        if r.returncode or not (Path(d) / "main.pdf").exists():
            print(r.stdout[-2000:], r.stderr[-2000:])
            return 1
    print("test compile from the unpacked package: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
