# Machine Learning Final Project

Wright State University, Fall 2026.

## Layout

- `paper/` — IEEE conference paper (LaTeX, `IEEEtran`), matching `docs/paper_template-finalProject.docx`
- `src/` — model and training code
- `notebooks/` — EDA and experiments
- `data/` — local datasets (git-ignored)
- `docs/` — course-provided templates and handouts

## Building the paper

```sh
cd paper && latexmk -pdf main.tex
```

Or upload the `paper/` folder to Overleaf.
