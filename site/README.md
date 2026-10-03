# ScopeCompose — project site

A static site that runs the detector rather than describing it. No build step,
no framework, no dependencies: open it and it works.

```
site/
  index.html                     markup
  favicon.svg
  assets/css/styles.css
  assets/js/engines.js           pure logic, ported from Python
  assets/js/app.js               DOM wiring
  data/corpus.js                 generated — what the page loads
  data/corpus.json               generated — the same records as plain JSON
  build_data.py                  regenerates both from the benchmark
  tools/make_reference.py        records Python's answers
  tools/verify_engines.js        replays them through the JavaScript
```

## Run it

Either open `site/index.html` in a browser, or serve it:

```bash
python -m http.server 8000 --directory site
# then http://localhost:8000
```

Both work. The corpus ships as `data/corpus.js` rather than a JSON file
fetched at runtime precisely so the first option works — `fetch` of a local
file is blocked by CORS on `file://`, and a demo that needs a web server is a
demo that fails in the room where it matters.

## Rebuild the data

Every figure on the page — composition counts, relation counts, the gold
branches the live engines run on — is read from the generated corpus, never
typed into the markup. After annotating or re-splitting:

```bash
python site/build_data.py
```

## The ported engine, and why it is checked

`assets/js/engines.js` reimplements `scope/attributes.py::compare`,
`ScopeAnalyser._combine`, `contract/routing.py::route` and `metrics/power.py`
in JavaScript, so the page computes its answers instead of displaying stored
ones.

A reimplementation can drift from the original, and a page that computes a
*different* relation from the pipeline is worse than a screenshot, because it
looks authoritative while being wrong. So the port is verified against Python
over the whole corpus:

```bash
python site/tools/make_reference.py    # record Python's answers
node   site/tools/verify_engines.js    # replay them through the JavaScript
```

Two checks run. `compare()` must match Python on every ordered branch pair in
both directions — including the SUBSET/SUPERSET asymmetry that decides which
branch is the default. And the four-way relation, derived live from the two
determinations, must equal the gold label on every two-branch instance.

Current status: **84/84 branch pairs match**, and **39/39 relations derived
match gold**.

Regenerate the reference whenever the corpus or `scope/attributes.py` changes.
`.github/workflows/pages.yml` runs the replay before every deploy, so a drifted
port stops the site shipping.

## Deploying

Pushing to `master` runs the verification and publishes `site/` to GitHub
Pages. This needs Pages switched on once, by hand:

> **Settings → Pages → Build and deployment → Source: GitHub Actions**

Until that is done the workflow will run its checks and then fail at the
publish step, which is harmless.

**The site reproduces unpublished results.** Enabling Pages makes the corpus,
the findings and the decisive experiment publicly readable. That is a decision
to take deliberately with a supervisor, not a side effect of a push.

## Conventions

- No dependencies, no build, no framework. Everything is served as authored.
- Motion is `transform` and `opacity` only; nothing is pinned to the scroll.
- Content is visible with CSS alone — JavaScript only adds motion, so the page
  degrades to a readable document rather than a blank one.
- Colour is never the only carrier of meaning; deleted branches also carry a
  strike, and unmeasured figures a hatch.
- `prefers-reduced-motion` and `prefers-color-scheme` are both honoured, and
  the theme toggle overrides the latter in either direction.
