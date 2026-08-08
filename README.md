# Project Website

The site is self-contained static HTML, CSS, JavaScript, and raster figures.
Preview it from the repository root:

```bash
python -m http.server 8765
```

Then open `http://localhost:8765/website/`. No build step or external runtime
dependency is required.

The source stays under `website/` on `main`. The deployed site is the root of
the separate `gh-pages` branch, so that branch contains only deployable site
files rather than training code, command manifests, or dataset labels. Publish
later website updates from a clean `main` checkout:

```bash
git subtree push --prefix website origin gh-pages
```
