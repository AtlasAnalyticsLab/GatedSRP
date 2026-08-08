# Project Website

The site is self-contained static HTML, CSS, JavaScript, and raster figures.
Preview it from the repository root:

```bash
python -m http.server 8765
```

Then open `http://localhost:8765/website/`. No build step or external runtime
dependency is required. For GitHub Pages, publish the `website/` directory as
the site root while keeping its `static/` paths unchanged.
