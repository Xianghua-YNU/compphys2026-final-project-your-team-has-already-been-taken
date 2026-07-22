# LaTeX compile notes

This template uses BibTeX, not biber.

Manual full build:

```powershell
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
```

Faster local builds:

```powershell
.\build.ps1 fast
```

`fast` writes `main-fast.pdf` and replaces the 72-frame animated TikZ figure with a single frame. Use this while editing text.

Final build:

```powershell
.\build.ps1 full -Bib
```

`full` writes `main.pdf` and keeps the complete animation. `-Bib` forces BibTeX; without it, the script reruns BibTeX only when the bibliography inputs look newer than the `.bbl`.
