# Website Migration Plan

## Architecture
Documenter.jl → Catppuccin Mocha → custom.css (sidebar restyle + website colors) + nav-inject.js (replace Documenter nav with website nav) → output directly to `website/MORFE website/docs/`

No post-processing. No template hacking. All customization via Documenter's `assets` mechanism (CSS + JS loaded at build time).

---

## Phase 1: Anonymize Website Content

### 1.1 `index.html`
- Specific performance numbers → `"{X}"`, `"{N}"`, `"Option A vs Option B"`
- Keep bar charts and equation strip visuals, replace numerical axis labels with generics
- Version 0.4/0.4.2 → **3.0.0**
- BibTeX block → `"Citation details coming with first publication"`
- Add **Current Status** section: describes MORFE.jl as active research code with low-level API
  (NDOrderModel, eigenproblems, solve_cohomological_problem)

### 1.2 `features.html`
- Benchmark chart shapes kept, axis labels → `"Method A"`, `"Method B"`
- `"vs POD-DEIM"` → `"vs Method A"`
- `"1000x faster"`, `"0.01% error"` → `"X times faster"`, `"Y% error"`
- `"does X"` → `"aims to do X"`
- ✅ badges on **real** features (list from src/MORFE.jl exports lines 41-83)
- 🚧 badges on **aspirational** features (parametrise, FEModel, Harmonic, continuation, reconstruct, plot, FRF)

### 1.3 `gallery.html`
- All cards kept, 📝 **"Planned"** overlay on each
- Description text → generic aspirational descriptions

### 1.4 `tutorials.html`
- Featured tutorial → `demo/ParametrisationMethod/demo_parametrisation_method.jl`
- Add Ferrite beam card → `demo/Ferrite/demo_mechanical_problem.jl`
- Add Gridap beam card → `demo/Gridap/demo_mechanical_problem.jl`
- Remove fictional micromirror link (covered by separate page)

### 1.5 `tutorial-micromirror.html`
- Keep all interactive visualizations intact
- Specific values → `"{displacement}"`, `"{frequency}"`, `"{error}%"`
- Comparison tables → `"Method A"`, `"Method B"`
- Replace fictional API calls (`parametrise(...)`) with real API calls from demo scripts

### 1.6 `docs.html`
- Sidebar restructured into two sections:
  - **Current API**: links to `docs/api/{Module}.html` for each real module
  - **Roadmap**: 🚧 aspirational features with GitHub issue links where applicable

---

## Phase 2: Documenter Configuration

### 2.1 `docs/make.jl`
```julia
makedocs(
    modules = [MORFE, ...],
    format = Documenter.HTML(
        prettyurls = false,
        assets = [
            "assets/custom.css",       # restyle sidebar + colors
            "assets/nav-inject.js",    # replace nav bar
        ],
        sidebar_sitename = false,      # hide "MORFE.jl" below logo in sidebar
        collapselevel = 1,             # flat sidebar (just module names)
        mathengine = KaTeX,            # or MathJax3
        footer = "MORFE.jl — Direct Parametrisation of Invariant Manifolds",
    ),
    build = "website/MORFE website/docs",  # output into website tree
    pages = [
        "Home" => "index.md",
        "API" => [
            "api/Multiindices.md",
            ...
        ],
        "Theory" => [...],
        "Tutorials" => [...],
    ],
)
```

### 2.2 `docs/src/assets/custom.css` — Full Rewrite

Targets: fonts, colors, sidebar appearance, code blocks, admonitions.

```css
/* Comment: This file restyles Documenter's HTML output (v1.x) to match the
   MORFE.jl website design. Depends on Documenter's class names. */

/* Website fonts */
body { font-family: 'Space Grotesk', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace; }

/* Website color palette — override Catppuccin Mocha */
:root {
  --jl-purple: #7c3aed;
  --jl-purple-light: #a78bfa;
  --jl-purple-dark: #5b21b6;
  --bg-primary: #0c0e17;
  --bg-secondary: #131626;
  --bg-elevated: #1a1d2e;
  --text-primary: #e8e8e8;
  --text-secondary: #8b8fa3;
  --border-color: #2a2d3a;
}

/* Sidebar — dark website aesthetic */
.docs-sidebar {
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-color);
}
.docs-sidebar .logo { filter: brightness(1.2); }
.docs-sidebar a,
.docs-sidebar .a { color: var(--text-secondary); }
.docs-sidebar a:hover,
.docs-sidebar .a:hover { color: var(--jl-purple); }
.docs-sidebar .current a,
.docs-sidebar .current .a { color: var(--jl-purple-light); }
.docs-sidebar li > ul { border-left-color: var(--border-color); }

/* Main content area */
.docs-main { background: var(--bg-primary); }
article { color: var(--text-primary); }
article h1, article h2, article h3, article h4 { color: var(--text-primary); }
article h1 { border-bottom-color: var(--border-color); }
article hr { border-color: var(--border-color); }

/* Code blocks */
code {
  background: var(--bg-elevated);
  color: #e2e8f0;
  padding: 0.15em 0.3em;
  border-radius: 4px;
}
pre { background: var(--bg-elevated); border: 1px solid var(--border-color); }
pre code { background: transparent; padding: 0; }

/* Docstring signatures */
.docs-sig-h1 { color: var(--jl-purple); }
.docs-sig { color: var(--text-secondary); }

/* Admonitions */
.admonition {
  background: var(--bg-elevated);
  border-color: var(--border-color);
  border-left-width: 4px;
}
.admonition.info { border-left-color: var(--jl-purple); }
.admonition.warning { border-left-color: #f59e0b; }
.admonition.danger { border-left-color: #ef4444; }
.admonition-title { color: var(--text-primary); }

/* Tables */
article table th { background: var(--bg-elevated); }
article table td, article table th { border-color: var(--border-color); }
article table tr:nth-child(even) { background: rgba(255,255,255,0.03); }

/* Links */
article a { color: var(--jl-purple-light); }
article a:hover { color: var(--jl-purple); }

/* Settings gear icon */
#documenter .docs-settings-button { color: var(--text-secondary); }
#documenter .docs-settings-button:hover { color: var(--text-primary); }

/* Catppuccin-specific overrides */
.theme--catppuccin-mocha {
  --brand: var(--jl-purple);
  --link-color: var(--jl-purple-light);
}
```

### 2.3 `docs/src/assets/nav-inject.js`

Replaces Documenter's top navigation bar with the website's nav on page load:

```javascript
document.addEventListener('DOMContentLoaded', function() {
  var nav = document.querySelector('nav');
  if (!nav) return;

  // Determine depth for relative links
  var path = location.pathname;
  var parts = path.replace(/\/$/, '').split('/');
  var depth = parts.filter(function(p) { return p.length > 0 && p !== 'index.html' && !p.endsWith('.html'); }).length;
  var prefix = depth > 0 ? '../'.repeat(depth) : './';

  nav.innerHTML = [
    '<a href="' + prefix + 'index.html" class="nav-logo">',
    '  <img src="' + prefix + 'assets/logo.svg" alt="MORFE.jl" height="36">',
    '</a>',
    '<div class="nav-links">',
    '  <a href="' + prefix + 'features.html">Features</a>',
    '  <a href="' + prefix + 'gallery.html">Gallery</a>',
    '  <a href="' + prefix + 'tutorials.html">Tutorials</a>',
    '  <a href="' + prefix + 'docs.html" class="active">Docs</a>',
    '</div>',
  ].join('');
  nav.className = 'website-nav';
});
```

### 2.4 Build output mapping

| Documenter source | Output path |
|---|---|
| `docs/src/index.md` | `website/MORFE website/docs/index.html` |
| `docs/src/api/Multiindices.md` | `website/MORFE website/docs/api/Multiindices.html` |
| `docs/src/theory/dpim.md` | `website/MORFE website/docs/theory/dpim.html` |
| `docs/src/tutorials/basic.md` | `website/MORFE website/docs/tutorials/basic.html` |
| `docs/src/assets/custom.css` | `website/MORFE website/docs/assets/custom.css` |
| `docs/src/assets/nav-inject.js` | `website/MORFE website/docs/assets/nav-inject.js` |

---

## Phase 3: Build

```bash
julia --project=docs docs/make.jl
# Output: website/MORFE website/docs/ with full Documenter tree
```

Open `website/MORFE website/index.html` and `website/MORFE website/docs/api/Multiindices.html` side by side to verify.

---

## Phase 4: Verification Checklist

### Content (4-1 to 4-6)
- [ ] `index.html`: no specific fabricated numbers, version is 3.0.0, Current Status section present
- [ ] `features.html`: real features = ✅, aspirational = 🚧, "aims to do" language, charts kept with generic labels
- [ ] `gallery.html`: all cards have 📝 Planned overlay
- [ ] `tutorials.html`: links point to real demos
- [ ] `tutorial-micromirror.html`: interactive visuals kept, data anonymized, real API calls
- [ ] `docs.html`: Current API + Roadmap sidebar sections

### Design (4-7 to 4-12)
- [ ] Sidebar uses website color palette (dark bg, purple accents, correct fonts)
- [ ] Article area has `--bg-primary` background, not white
- [ ] Fonts are Space Grotesk (body) and JetBrains Mono (code)
- [ ] Code blocks have dark `--bg-elevated` background
- [ ] Admonitions match dark theme
- [ ] Website nav replaces Documenter nav, links work correctly at all depth levels

### Functionality (4-13 to 4-17)
- [ ] Documenter search works and is styled consistently
- [ ] KaTeX math renders correctly
- [ ] Cross-references between pages work
- [ ] All `.html` links in the docs directory resolve (no 404s)
- [ ] `dpim-preview.html` — unchanged, still works

---

## Files Modified

| File | Change type | Description |
|------|-------------|-------------|
| `website/MORFE website/index.html` | Edit | Anonymize claims, add Current Status, fix version |
| `website/MORFE website/features.html` | Edit | ✅/🚧 badges, anonymize, "aims to do" |
| `website/MORFE website/gallery.html` | Edit | 📝 Planned overlays |
| `website/MORFE website/tutorials.html` | Edit | Real demo links |
| `website/MORFE website/tutorial-micromirror.html` | Edit | Anonymize data, real API |
| `website/MORFE website/docs.html` | Edit | Restructure sidebar |
| `docs/src/assets/custom.css` | Rewrite | Full rewrite for dark website theme |
| `docs/src/assets/nav-inject.js` | **New** | Replace Documenter nav with website nav |
| `docs/make.jl` | Edit | Set build path, add nav-inject.js to assets |
| All `docs/src/api/*.md` (16 files) | Verify | May need updates for completeness |

## Files Unchanged

| File | Reason |
|------|--------|
| `website/MORFE website/assets/site.css` | Source of truth for website styles |
| `website/MORFE website/assets/docs.css` | May need later removal/archiving |
| `website/MORFE website/assets/manifold-bg.js` | Visual effect, no changes needed |
| `website/MORFE website/dpim-preview.html` | Valid standalone math visualization |
| All `src/*.jl` | Real codebase, no changes |
| All `demo/*.jl` | Real demos, no changes |
| `Project.toml` | No changes needed |

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Documenter changes CSS class names in future version | Sidebar styling breaks | Pin Documenter version in docs/Project.toml |
| `nav-inject.js` race condition | Nav not replaced | Use `setTimeout` or observe nav availability |
| Catppuccin Mocha theme upgrade overrides custom CSS | Colors revert | Put `!important` on key overrides, test after upgrade |
| Relative link depth calculation fails for edge-case URLs | Nav links broken | Test with `file://` and served via `python -m http.server` |
| Search index path wrong after moving build output | Search returns 404s | Verify `assets/search_index.json` relative path |