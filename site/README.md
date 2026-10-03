# ScopeCompose — project site

A static site that runs the detector rather than describing it. No build step,
no framework, no dependencies: open it and it works.

```
site/
  index.html                     markup
  favicon.svg
  assets/css/styles.css
  assets/js/engines.js           scope algebra, four-way relation, routing
  assets/js/match.js             branch matching and the textual judge
  assets/js/detector.js          lexical features, heuristic NLI, stage 1
  assets/js/compose.js           composition operator and renderer
  assets/js/stats.js             Wilson, McNemar, Holm-Bonferroni
  assets/js/motion.js            WebGL field, reveals, cursor, intro
  assets/js/app.js               overview wiring
  assets/js/bench.js             the bench, and hash routing
  assets/js/byo.js               bring your own branches
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

## Routes

| route | what it is |
|---|---|
| `#/` | the overview: the live demo, the four relations, findings, claims |
| `#/bench` | five resolvers scored side by side, the detector trace, retrieval loss, the order-invariance prover |
| `#/byo` | type two branches and run the composition operator on them |

Anything that is not `#/...` falls through to the overview, so in-page
anchors like `#relations` keep working.

**The bench is the one to show someone.** It answers "how is this different
from what already exists?" by running the alternatives rather than describing
them, and the answer it gives is not a win:

```
Selection (prior work)             PR 0.542
ScopeCompose, as it actually runs  PR 0.446   routing-level
Full context, both passages        PR 0.855   gameable
ScopeCompose, routing oracle       PR 0.976   ORACLE, upper bound
Gold scoped answer                 PR 0.699   the scorer's ceiling
```

The composition operator preserves 97.6% of gold branches when it is told a
conflict exists. The detector in front of it finds **0 of 42**, topping out at
0.2206 against an escalation floor of 0.35 — so stage 2 never runs, with or
without an API key. That gap is the remaining research problem, and it is the
opposite of the one the project expected.

Three framing rules the bench must not lose:

- The oracle column is fed gold routing **and** gold branches, because B1
  extraction has no offline path. It carries a permanent badge.
- The as-it-runs column is scored at the routing step, exactly as
  `run_decisive` scores the pipeline arm, so it reads no gold branch text.
  An earlier version fed gold branches into `compose()` and then scored the
  rendered string against those same branches, which inflated it to 0.687.
- Concatenation scoring above the gold answer is a defect in the metric, not
  a property of the resolver: PR is containment with no length normalisation
  and no precision counterweight. It is shown with answer length beside it.

## The motion layer

`assets/js/motion.js` carries the WebGL field, the word reveals, the cursor
and the intro. Three rules it does not break:

- **Native scrolling is never intercepted.** No scroll-jacking, nothing
  pinned, no smoothing library. On a high-refresh display the browser's own
  scrolling is smoother than anything that replaces it, and every scroll
  handler here is passive and writes a custom property rather than reading
  layout.
- **GPU or nothing.** The field is one fullscreen fragment shader at a capped
  1.5x DPR, paused by `IntersectionObserver` the moment it leaves view.
  Everything else is `transform` and `opacity`.
- **The field means something.** The two regions are the two applicability
  sets. Their centres and radii come from whatever `compare()` returned for
  the pair on screen, and the contour drawn through each is the applicability
  boundary. Change the instance in the hero and the geometry changes with it.

If WebGL is unavailable, `app.js` falls back to a 2D canvas. `<html>` carries
`gl` or `no-gl` so which path ran is observable rather than guessed.

**The intro veil is fail-safe.** It is lifted by JavaScript adding `.ready`,
but a CSS animation drops it at 2.8s regardless. A veil that depends solely on
a script is a permanently blank page the first time that script fails, which
is not a risk worth taking with a page shown to an audience.

> Verifying this with headless Chrome is limited: screenshots there do not
> advance CSS transitions, and WebGL is not composited into them at all. The
> engine output was checked instead by reading pixels back out of the GL
> framebuffer, and the reveal and layout behaviour through DOM assertions.

## Conventions

- No dependencies, no build, no framework. Everything is served as authored.
- Motion is `transform` and `opacity` only; nothing is pinned to the scroll.
- Content is visible with CSS alone — JavaScript only adds motion, so the page
  degrades to a readable document rather than a blank one.
- Colour is never the only carrier of meaning; deleted branches also carry a
  strike, and unmeasured figures a hatch.
- `prefers-reduced-motion` and `prefers-color-scheme` are both honoured, and
  the theme toggle overrides the latter in either direction.
