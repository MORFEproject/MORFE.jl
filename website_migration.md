# Website Migration Plan

## Purpose

The website serves two audiences simultaneously:

- **External**: shows what MORFE.jl does, how to install it, and how to use it — today and in the near future.
- **Internal (developers)**: functions as a north star that makes the development goal tangible. The target high-level API (`parametrise`, `FEModel`, `Harmonic`, `continuation`) is shown openly as the design contract that the low-level implementation is being built to satisfy.

The aspirational API snippets are **executable specifications** — they define the interface surface that the current `solve_cohomological_problem` / `NDOrderModel` layer will eventually be wrapped into.

---

## Architecture

One script does the work (the two-script JSON intermediate approach was simplified to avoid adding a JSON library dependency):

| Script | Input | Output |
|--------|-------|--------|
| `scripts/generate_api.jl` ✅ | MORFE source (loaded via `Base.Docs`) | `website/MORFE website/api.html` |

No Documenter.jl. No CSS override fights. `api.html` is a plain website page that shares `site.css`, `docs.css`, the same fonts, palette, and nav as every other page.

The existing `docs/` folder and `docs/make.jl` are **retained for CI doctest only** — the built HTML output is no longer used by the website.

---

## Navigation

### Current (7 links — too many)

```
Overview · Features · Gallery · Tutorials · Docs · Publications · Team  [GitHub]
```

### Target (5 links)

```
Overview · Features · Tutorials · API · Publications  [GitHub]
```

| Removed item | Disposition |
|---|---|
| Gallery | Accessible from Features page (add a "Examples" anchor section); remove from main nav |
| Docs | Retired; replaced by `api.html` |
| Team | Content moved to the bottom of `index.html` (new "Team" section); `team.html` kept as a standalone page but not in the main nav |

Every `.html` file and `nav-inject.js` must be updated to use the new 5-link nav.

### Nav HTML block (copy-paste target for all pages)

```html
<div class="nav-links">
  <a href="index.html">Overview</a>
  <a href="features.html">Features</a>
  <a href="tutorials.html">Tutorials</a>
  <a href="api.html">API</a>
  <a href="publications.html">Publications</a>
</div>
```

Each page sets `class="active"` on its own link.

---

## Status: what is already done

The following phases from the previous plan are **complete**:

| Item | State |
|------|-------|
| `docs/make.jl` — build path, `nav-inject.js`, `sidebar_sitename=false` | ✅ done |
| `docs/src/assets/custom.css` — dark theme rewrite | ✅ done |
| `docs/src/assets/nav-inject.js` — DOMContentLoaded-safe injection, 5-link nav | ✅ done |
| `docs/src/api/eigenproblems.md` and `invariance_error.md` (new files) | ✅ done |
| `index.html` — anonymised numbers, removed version string, Examples + Team sections | ✅ done |
| `features.html` — ✅/🚧 badges, generic benchmark labels, implementation status table | ✅ done |
| `gallery.html` — 🚧 Roadmap overlay on all 8 cards | ✅ done |
| `tutorials.html` — real demo links, demo cards, 🚧 on micromirror | ✅ done |
| `tutorial-micromirror.html` — anonymised values, `# target API` annotations | ✅ done |
| `scripts/generate_api.jl` — single-pass extraction + generation, 113 entries across 16 modules | ✅ done |
| `website/MORFE website/api.html` — generated (163 KiB) | ✅ done |
| Nav updated to 5 links on all 8 HTML files + `nav-inject.js` | ✅ done |
| `docs.html` — replaced with redirect to `api.html` | ✅ done |

All phases are complete.

---

## Phase A — Docstring extraction (`scripts/extract_docs.jl`)

A Julia script that loads MORFE and walks every submodule using `Base.Docs` introspection.
Does not depend on Documenter.jl at all.

### Modules to document (mirrors the `makedocs` list)

```julia
const MODULES = [
    (MORFE.Multiindices,             "Multiindices"),
    (MORFE.Polynomials,              "Polynomials"),
    (MORFE.MultilinearMaps,          "MultilinearMaps"),
    (MORFE.ExternalSystems,          "ExternalSystems"),
    (MORFE.FullOrderModel,           "FullOrderModel"),
    (MORFE.Eigenproblems,            "Eigenproblems"),
    (MORFE.Eigensolvers,             "Eigensolvers"),
    (MORFE.JordanChain,              "JordanChain"),
    (MORFE.PropagateEigenmodes,      "PropagateEigenmodes"),
    (MORFE.Realification,            "Realification"),
    (MORFE.Resonance,                "Resonance"),
    (MORFE.InvarianceEquation,       "InvarianceEquation"),
    (MORFE.MasterModeOrthogonality,  "MasterModeOrthogonality"),
    (MORFE.ParametrisationMethod,    "ParametrisationMethod"),
    (MORFE.MultilinearTerms,         "MultilinearTerms"),
    (MORFE.LowerOrderCouplings,      "LowerOrderCouplings"),
    (MORFE.CohomologicalEquations,   "CohomologicalEquations"),
    (MORFE.InvarianceError,          "InvarianceError"),
]
```

### Extraction logic

For each module:
1. `names(mod, all=true)` — get every symbol defined in that module
2. Filter to symbols whose `parentmodule` is the module itself (exclude re-exports)
3. For each symbol, call `Base.Docs.doc(Base.Docs.Binding(mod, sym))` to retrieve the `Markdown.MD` object
4. Skip symbols with no docstring (`isa(doc, Markdown.MD) && isempty(doc.content)`)
5. Determine `kind`: `"type"` if the binding resolves to a `DataType` or `UnionAll`, `"macro"` if the name starts with `@`, `"function"` otherwise
6. Get the method signature list via `methods(getfield(mod, sym))` where applicable
7. Convert the `Markdown.MD` docstring to HTML via `Markdown.html(doc)`

### Output schema (`scripts/docs_data.json`)

```json
{
  "generated": "2026-05-16T...",
  "modules": [
    {
      "name": "Multiindices",
      "items": [
        {
          "name": "MultiindexSet",
          "kind": "type",
          "signatures": ["MultiindexSet{NVAR}"],
          "doc_html": "<p>Graded-lex ordered set of <code>SVector{NVAR,Int}</code> exponents...</p>"
        },
        {
          "name": "grlex_index",
          "kind": "function",
          "signatures": [
            "grlex_index(α::SVector{NVAR,Int}) → Int",
            "grlex_index(nvar, order) → Int"
          ],
          "doc_html": "<p>Return the 1-based graded-lex index of multiindex <code>α</code>.</p>"
        }
      ]
    }
  ]
}
```

---

## Phase B — API page generator (`scripts/generate_api.jl`)

Reads `scripts/docs_data.json`, writes `website/MORFE website/api.html`.

The generated page has the same `<head>`, nav, and CSS as every other marketing page.
It needs no JavaScript beyond what `site.css` already uses.

### Page layout

```
┌──────────────────────────────────────────────────────────────────┐
│  <nav>  (shared nav — 5 links)                                   │
├──────────────────────┬───────────────────────────────────────────┤
│  Module rail         │  Content area                            │
│  (sticky, ~200px)    │                                           │
│                      │  ## Multiindices                          │
│  • Multiindices      │                                           │
│  • Polynomials       │  ### MultiindexSet    [type]              │
│  • MultilinearMaps   │  ┌─────────────────────────────────────┐  │
│  • ...               │  │ MultiindexSet{NVAR}                 │  │
│                      │  │─────────────────────────────────────│  │
│  [search input]      │  │ Graded-lex ordered set of …         │  │
│                      │  └─────────────────────────────────────┘  │
│                      │                                           │
│                      │  ### grlex_index    [function]            │
│                      │  ...                                      │
└──────────────────────┴───────────────────────────────────────────┘
```

### HTML structure of a docstring card

```html
<div class="api-entry" id="Multiindices-MultiindexSet">
  <div class="api-entry-header">
    <code class="api-name">MultiindexSet{NVAR}</code>
    <span class="api-kind">type</span>
  </div>
  <div class="api-sigs">
    <code>MultiindexSet{NVAR}</code>
  </div>
  <div class="api-doc">
    <!-- doc_html content verbatim -->
  </div>
</div>
```

### CSS additions needed in `site.css` (or a new `api.css`)

```css
/* Module rail */
.api-rail {
  position: sticky;
  top: calc(var(--nav-h) + 24px);
  width: 200px;
  flex-shrink: 0;
  font-family: var(--mono);
  font-size: 13px;
}
.api-rail a {
  display: block;
  padding: 5px 10px;
  color: var(--ink-3);
  border-radius: 4px;
  text-decoration: none;
}
.api-rail a:hover, .api-rail a.active { color: var(--ink); background: var(--bg-3); }

/* Search filter */
.api-search {
  width: 100%;
  background: var(--bg-3);
  border: 1px solid var(--hair-2);
  color: var(--ink);
  font-family: var(--mono);
  font-size: 13px;
  padding: 6px 10px;
  border-radius: 4px;
  margin-bottom: 16px;
}

/* Docstring card */
.api-entry {
  border: 1px solid var(--hair);
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  background: var(--bg-2);
  padding: 1rem 1.2rem;
  margin-bottom: 1.2rem;
}
.api-entry-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 0.5rem;
}
.api-name {
  font-family: var(--mono);
  font-size: 1rem;
  font-weight: 600;
  color: var(--accent);
}
.api-kind {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
  border: 1px solid var(--hair-2);
  border-radius: 3px;
  padding: 1px 6px;
}
.api-sigs {
  margin-bottom: 0.8rem;
}
.api-sigs code {
  display: block;
  font-size: 0.875rem;
  color: var(--ink-2);
  background: transparent;
  padding: 0;
}
.api-doc { font-size: 0.95rem; line-height: 1.65; color: var(--ink-2); }
.api-doc p { margin: 0.4rem 0; }
.api-doc code { font-family: var(--mono); font-size: 0.875em; background: var(--bg-3); padding: 0.1em 0.3em; border-radius: 3px; }
.api-doc pre { background: #07070b; border: 1px solid var(--hair); border-radius: 6px; padding: 0.8rem 1rem; overflow-x: auto; }
.api-doc pre code { background: transparent; padding: 0; }

/* Section heading per module */
.api-module-heading {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--ink);
  border-bottom: 1px solid var(--hair);
  padding-bottom: 0.3rem;
  margin: 2.5rem 0 1.2rem;
  scroll-margin-top: calc(var(--nav-h) + 16px);
}

/* Live filter: hide non-matching entries */
.api-entry[hidden] { display: none; }
```

### Client-side search (inline `<script>` at bottom of `api.html`)

```javascript
var input = document.getElementById('api-search');
var entries = document.querySelectorAll('.api-entry');
input.addEventListener('input', function () {
  var q = this.value.toLowerCase();
  entries.forEach(function (e) {
    var match = !q || e.textContent.toLowerCase().includes(q);
    e.hidden = !match;
  });
});
```

---

## Phase C — Navigation updates

Every `.html` file in `website/MORFE website/` and `nav-inject.js` must be updated to the new 5-link nav.

### Files to update

| File | Active link | Notes |
|------|-------------|-------|
| `index.html` | Overview | Also: add Team section at bottom (see Phase D) |
| `features.html` | Features | Also: add Examples anchor section (Gallery content) |
| `gallery.html` | — | No active link; page stays for direct links, not in nav |
| `tutorials.html` | Tutorials | |
| `tutorial-micromirror.html` | Tutorials | |
| `docs.html` | — | Retire: redirect `<meta http-equiv="refresh" content="0; url=api.html">` |
| `publications.html` | Publications | |
| `team.html` | — | No active link; page stays for direct links, not in nav |
| `api.html` | API | New generated file |
| `docs/src/assets/nav-inject.js` | API (active) | Used on all Documenter pages (CI only) |

### Nav block for `nav-inject.js`

```javascript
'<div class="nav-links">' +
  '<a href="' + siteRoot + 'index.html">Overview</a>' +
  '<a href="' + siteRoot + 'features.html">Features</a>' +
  '<a href="' + siteRoot + 'tutorials.html">Tutorials</a>' +
  '<a href="' + siteRoot + 'api.html" class="active">API</a>' +
  '<a href="' + siteRoot + 'publications.html">Publications</a>' +
'</div>'
```

---

## Phase D — `index.html` additions

Two new sections appended before `</main>` or equivalent:

### D.1 — Examples teaser (replaces Gallery in nav)

A 2–3 card row linking to the demo scripts, styled like the existing tutorial cards but compact. Anchor: `#examples`.

```html
<section id="examples" class="section">
  <h2>Examples</h2>
  <div class="card-row">
    <!-- card: demo/ParametrisationMethod/ -->
    <!-- card: demo/Ferrite/ -->
    <!-- card: demo/Gridap/ -->
  </div>
  <p class="muted">Full gallery coming soon.</p>
</section>
```

### D.2 — Team section (replaces `team.html` in nav)

Copy the content from `team.html`'s main section verbatim. Anchor: `#team`. The standalone `team.html` keeps working for direct links.

---

## Phase E — `docs.html` retirement

Replace the full page content with a redirect:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="0; url=api.html" />
<title>Redirecting — MORFE.jl</title>
</head>
<body>
<p><a href="api.html">Redirecting to API reference…</a></p>
</body>
</html>
```

---

## Regeneration workflow

```bash
# From repo root — run after any docstring change:
julia --project scripts/extract_docs.jl    # → scripts/docs_data.json
julia --project scripts/generate_api.jl   # → website/MORFE website/api.html
```

No web server needed. `api.html` opens correctly via `file://` because it has no
relative-depth path calculations.

To serve locally:
```bash
cd "website/MORFE website"
python3 -m http.server 8080
# http://localhost:8080/api.html
# http://localhost:8080/index.html
```

---

## Files to create

| File | Description |
|------|-------------|
| `scripts/extract_docs.jl` | Julia: loads MORFE, walks `Base.Docs`, writes `docs_data.json` |
| `scripts/generate_api.jl` | Julia: reads `docs_data.json`, writes `api.html` |
| `website/MORFE website/api.html` | Generated output — do not edit by hand |

## Files to modify

| File | Change |
|------|--------|
| `website/MORFE website/index.html` | Update nav; add Examples section (D.1); add Team section (D.2) |
| `website/MORFE website/features.html` | Update nav; add Examples anchor section |
| `website/MORFE website/gallery.html` | Update nav (no active link) |
| `website/MORFE website/tutorials.html` | Update nav |
| `website/MORFE website/tutorial-micromirror.html` | Update nav |
| `website/MORFE website/publications.html` | Update nav |
| `website/MORFE website/team.html` | Update nav (no active link) |
| `website/MORFE website/docs.html` | Replace with redirect to `api.html` |
| `docs/src/assets/nav-inject.js` | Update nav links (used in CI Documenter output only) |
| `website/MORFE website/assets/site.css` | Add API-page CSS (`.api-entry`, `.api-rail`, `.api-search`, `.api-module-heading`) |

## Files unchanged

| File | Reason |
|------|--------|
| `docs/make.jl` | Kept for CI doctest; HTML output no longer used by website |
| `docs/src/assets/custom.css` | Still used in CI Documenter output |
| `website/MORFE website/assets/manifold-bg.js` | Visual effect, no changes |
| `website/MORFE website/dpim-preview.html` | Standalone visualization |
| All `src/*.jl`, `demo/*.jl` | Source code |

---

## Verification checklist

### API page
- [ ] `api.html` opens without a server and styles correctly
- [ ] All 18 modules appear as sections
- [ ] Each entry has: name, kind badge, signatures, docstring HTML
- [ ] Entries with no docstring are omitted (not shown as blank cards)
- [ ] Search input filters entries live as you type
- [ ] Module rail anchor links scroll to correct section
- [ ] Math in docstrings renders (KaTeX loaded in `api.html` `<head>`)
- [ ] Code blocks in docstrings are styled with `#07070b` background

### Navigation
- [ ] All 8 HTML files have the 5-link nav
- [ ] No link 404s (test each of the 5 links from each page)
- [ ] `docs.html` redirects to `api.html`
- [ ] `gallery.html` and `team.html` are reachable via direct link but absent from nav
- [ ] `index.html` has working Examples (`#examples`) and Team (`#team`) anchor sections

### Content
- [ ] `index.html` Team section matches `team.html` content
- [ ] Features page has an Examples section linking to the three demo scripts
