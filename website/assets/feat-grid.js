/**
 * feat-grid.js — interactive capabilities grid for MORFE.jl
 *
 * Usage: add <div id="feat-grid-mount"></div> in the page and
 *        <script src="assets/feat-grid.js"></script> in <head>.
 *
 * To edit content, update the FEATS array below.
 * Each entry: { num, color, title, short, detail: { body, bullets[], link: { href, label } } }
 *
 * Interaction:
 *   Click a card → it expands in-place (stays in its grid column, grows vertically).
 *   The row below slides down; nothing else reflows.
 *   Click the same card again → collapses.
 */
(function () {

  // ─── Feature data ────────────────────────────────────────────────────────
  // Edit only this array to update grid content.
  // tint: rgba overlay applied as background when the card is active.

  var KATEX_OPTS = {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$',  right: '$',  display: false }
    ]
  };

  var FEATS = [
    {
      num: 'F.01',
      color: '--jl-purple',
      tint: 'rgba(149,88,178,0.10)',
      title: 'Invariant Manifolds',
      short: 'An $n$-dimensional invariant manifold tangent to the chosen eigenspace — curved coordinates, not a Galerkin projection.',
      detail: {
        body: 'MORFE parametrises invariant manifolds tangent to chosen eigenspaces. The reduced ODE $\\dot{\\mathbf{z}} = \\mathbf{R}(\\mathbf{z}, \\mathbf{r})$ governs the reduced coordinates on the curved manifold.',
        bullets: [
          'Distinct from projection-based methods: no projection error',
          'Builds an asymptotic expansion order-by-order by solving cohomological equations',
          'Computes Nonlinear Normal Modes (NNMs)',
          'Computes Spectral Submanifolds (SSMs) provided special non-resonance conditions hold'
        ],
        link: { href: 'features.html', label: 'Method details →' }
      }
    },
    {
      num: 'F.02',
      color: '--jl-red',
      tint: 'rgba(203,60,51,0.10)',
      title: 'Large-scale FE Models',
      short: 'Reads sparse $\\mathbf{M}, \\mathbf{C}, \\mathbf{K}$ directly. Nonlinear terms assembled element-by-element — no full tensor formed. Tested at $N > 10^6$ DOFs.',
      detail: {
        body: 'Sparse $\\mathbf{M}, \\mathbf{C}, \\mathbf{K}$ are read directly from Ferrite.jl or Gridap.jl. Nonlinear terms implement <code>FEMMultilinearMap</code> — an element-level interface that lets MORFE.jl traverse the mesh once per monomial without assembling global tensors $\\mathbf{G}_{ijk}$ or $\\mathbf{H}_{ijkl}$.',
        bullets: [
          'Tested at $N > 10^6$ DOFs on a clamped-clamped beam',
          'Ferrite.jl and Gridap.jl backends included in <code>demo/</code>',
          '<code>fem_elements</code>, <code>scatter_qp!</code>, <code>accumulate_qp!</code> interface'
        ],
        link: { href: 'documentation.html', label: 'FEMMultilinearMap docs →' }
      }
    },
    {
      num: 'F.03',
      color: '--jl-green',
      tint: 'rgba(56,152,38,0.10)',
      title: 'Modal interactions',
      short: 'Arbitrary $p:q$ internal resonances detected from the spectrum and retained in the ROM automatically.',
      detail: {
        body: 'DPIM automatically detects internal resonances from the linearised spectrum. Any $p:q$ integer relation among master and forcing eigenvalues generates corresponding reduced dynamics in $\\mathbf{R}(\\mathbf{z}, \\mathbf{r})$.',
        bullets: [
          '1:2, 1:3, 5:3 and arbitrary combinatorial combinations',
          'Detection based on condition number with adjustable tolerance',
          'Resonant terms retained automatically in the reduced dynamics'
        ],
        link: { href: 'features.html#resonance', label: 'Resonance conditions →' }
      }
    },
    {
      num: 'F.04',
      color: '--jl-blue',
      tint: 'rgba(64,99,216,0.10)',
      title: 'Quasi-periodic &amp; chaotic forcing',
      short: 'External excitation is a small autonomous system which encodes single-frequency, multi-frequency quasi-periodic, or weakly chaotic input.',
      detail: {
        body: 'External forcing enters as an autonomous <code>ExternalSystem</code> $\\dot{\\mathbf{r}} = \\mathbf{E}(\\mathbf{r})$. Linear systems with eigenvalues $\\pm i\\Omega_k$ encode multi-frequency quasi-periodic input; nonlinear external systems encode weakly chaotic forcing — all without time-averaging.',
        bullets: [
          'Single-frequency: $\\mathbf{E} = \\mathrm{diag}(i\\Omega,\\,-i\\Omega)$',
          'Multi-frequency: stack multiple $\\pm i\\Omega_k$ pairs',
          'Forcing enters through <code>multiplicity_external</code> in <code>MultilinearMap</code>'
        ],
        link: { href: 'features.html#forcing', label: 'External system →' }
      }
    },
    {
      num: 'F.05',
      color: '--jl-purple',
      tint: 'rgba(149,88,178,0.10)',
      title: 'Multiphysics',
      short: 'Physics-agnostic interface: structural, electrostatic, MEMS, thermal. Any $p$-th order polynomial ODE on FE matrices is reducible.',
      detail: {
        body: '<code>NthOrderModel</code> accepts any $p$-th order polynomial ODE on sparse FE matrices. The FEM interface is physics-agnostic: the same pipeline reduces structural, electrostatic, piezoelectric, and thermal models.',
        bullets: [
          '<code>ORD</code> type parameter encodes ODE order at compile time',
          'Nonlinear terms defined by multiindex + user-supplied <code>f!</code>',
          'No physics assumptions hard-coded in the solver'
        ],
        link: { href: 'documentation.html', label: 'NthOrderModel →' }
      }
    },
    {
      num: 'F.06',
      color: '--jl-red',
      tint: 'rgba(203,60,51,0.10)',
      title: 'Julia ecosystem',
      short: 'Ferrite.jl, Gridap.jl, BifurcationKit, DifferentialEquations. Type-stable, multiple-dispatch. MIT licensed.',
      detail: {
        body: 'MORFE.jl is built on Julia\'s type-stable, multiple-dispatch core. It composes with Ferrite.jl, Gridap.jl, Arpack, BifurcationKit, and DifferentialEquations.jl out of the box.',
        bullets: [
          'MIT licensed, open-source on GitHub',
          '<code>FEMMultilinearMap</code> interface extensible to any Julia FEM library',
          'KLU / SuiteSparse sparse solvers auto-selected'
        ],
        link: { href: 'https://github.com/MORFEproject/MORFE.jl', label: 'GitHub →' }
      }
    }
  ];

  // ─── CSS ─────────────────────────────────────────────────────────────────
  // Injected inside DOMContentLoaded — appended last in <head>, wins over
  // the page's embedded <style> block at equal specificity.

  var CSS = [
    /* ── Summary (click target) ── */
    '.feat-summary {',
    '  display: block;',
    '  cursor: pointer;',
    '}',

    /* ── Chevron ── */
    '.feat-summary .feat-chevron {',
    '  position: absolute; bottom: 14px; right: 16px;',
    '  font-family: var(--mono); font-size: 16px; color: var(--ink-3);',
    '  line-height: 1;',
    '  transition: transform 0.3s ease, color 0.2s;',
    '  pointer-events: none; user-select: none;',
    '}',
    '.feat.is-active .feat-summary .feat-chevron {',
    '  transform: rotate(45deg);',
    '  color: var(--ink-2);',
    '}',

    /* ── Active card: highlight only — no grid-column change ── */
    /* The card stays in its column. The row grows vertically. Cards in the */
    /* row below slide down. Nothing else moves. */
    /* Background tint is applied via inline style in JS (colour-per-card). */
    '.feat.is-active {',
    '  transition: background 0.3s;',
    '}',
    '.feat.is-active .accent-bar {',
    '  width: 100%;',
    '  transition: width 0.4s cubic-bezier(0.4,0,0.2,1);',
    '}',

    /* ── Expand panel ── */
    /* margin: 0 -32px pulls the panel to the card's left/right inner edges */
    /* so the border-top divider runs edge-to-edge. Collapsed height = 0. */
    '.feat-expand-panel {',
    '  display: grid;',
    '  grid-template-rows: 0fr;',
    '  transition:',
    '    grid-template-rows 0.38s cubic-bezier(0.4,0,0.2,1),',
    '    margin-top         0.38s cubic-bezier(0.4,0,0.2,1);',
    '  margin: 0 -32px 0;',
    '}',
    '.feat.is-active .feat-expand-panel {',
    '  grid-template-rows: 1fr;',
    '  margin-top: 20px;',
    '}',
    '.feat-expand-inner {',
    '  overflow: hidden;',
    '  min-height: 0;',
    '}',

    /* ── Detail content: single-column, fits the 1/3-width card ── */
    '.feat-expand-body {',
    '  padding: 22px 32px 28px;',
    '  border-top: 1px solid var(--hair);',
    '}',
    '.feat-expand-text {',
    '  color: var(--ink-2);',
    '  font-size: 14px;',
    '  line-height: 1.7;',
    '  margin: 0 0 14px;',
    '}',
    '.feat-expand-bullets {',
    '  margin: 14px 0 14px;',
    '  padding-left: 16px;',
    '  color: var(--ink-2);',
    '  font-size: 14px;',
    '  line-height: 1.7;',
    '}',
    '.feat-expand-bullets li { margin-bottom: 5px; }',
    '.feat-expand-link {',
    '  display: inline-block;',
    '  font-family: var(--mono); font-size: 12px;',
    '  color: var(--ink-3); text-decoration: none;',
    '  border-bottom: 1px solid var(--hair); padding-bottom: 1px;',
    '  transition: color 0.2s, border-color 0.2s;',
    '}',
    '.feat-expand-link:hover { color: var(--ink); border-color: var(--hair-2); }'
  ].join('\n');

  // ─── HTML builders ────────────────────────────────────────────────────────

  function buildBullets(bullets) {
    if (!bullets || !bullets.length) { return ''; }
    var items = '';
    for (var i = 0; i < bullets.length; i++) {
      items += '<li>' + bullets[i] + '</li>';
    }
    return '<ul class="feat-expand-bullets">' + items + '</ul>';
  }

  function buildCard(f, idx) {
    var d = f.detail;
    var linkHTML = d.link
      ? '<a class="feat-expand-link" href="' + d.link.href + '">' + d.link.label + '</a>'
      : '';

    return (
      '<div class="feat" data-feat-idx="' + idx + '" aria-expanded="false">' +
        '<div class="accent-bar"></div>' +
        '<div class="feat-summary" tabindex="0" role="button">' +
          '<div class="num">' + f.num + '</div>' +
          '<h3>' + f.title + '</h3>' +
          '<p>' + f.short + '</p>' +
          '<span class="feat-chevron" aria-hidden="true">+</span>' +
        '</div>' +
        '<div class="feat-expand-panel">' +
          '<div class="feat-expand-inner">' +
            '<div class="feat-expand-body">' +
              '<p class="feat-expand-text">' + d.body + '</p>' +
              buildBullets(d.bullets) +
              linkHTML +
            '</div>' +
          '</div>' +
        '</div>' +
      '</div>'
    );
  }

  function buildGrid() {
    var html = '<div class="feat-grid">';
    for (var i = 0; i < FEATS.length; i++) {
      html += buildCard(FEATS[i], i);
    }
    return html + '</div>';
  }

  // ─── Interaction ──────────────────────────────────────────────────────────

  function mount(mountEl) {
    mountEl.innerHTML = buildGrid();

    var cards = mountEl.querySelectorAll('.feat');
    var activeIdx = -1;

    function openCard(idx) {
      for (var i = 0; i < cards.length; i++) {
        var active = (i === idx);
        cards[i].classList.toggle('is-active', active);
        cards[i].setAttribute('aria-expanded', active ? 'true' : 'false');
        cards[i].style.background = active ? FEATS[i].tint : '';
      }
      activeIdx = idx;
    }

    function closeCard() {
      for (var i = 0; i < cards.length; i++) {
        cards[i].classList.remove('is-active');
        cards[i].setAttribute('aria-expanded', 'false');
        cards[i].style.background = '';
      }
      activeIdx = -1;
    }

    for (var i = 0; i < cards.length; i++) {
      (function (idx) {
        var summary = cards[idx].querySelector('.feat-summary');

        summary.addEventListener('click', function () {
          if (activeIdx === idx) { closeCard(); } else { openCard(idx); }
        });

        summary.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (activeIdx === idx) { closeCard(); } else { openCard(idx); }
          }
        });
      }(i));
    }
  }

  // ─── Init ─────────────────────────────────────────────────────────────────

  function renderKaTeX(el) {
    if (window.renderMathInElement) {
      window.renderMathInElement(el, KATEX_OPTS);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    var s = document.createElement('style');
    s.textContent = CSS;
    document.head.appendChild(s);

    var el = document.getElementById('feat-grid-mount');
    if (!el) { return; }
    mount(el);
    // Defer scripts (katex + auto-render) execute before DOMContentLoaded, so
    // renderMathInElement(document.body) already fired against an empty mount
    // point. Re-render the injected content explicitly here.
    renderKaTeX(el);
    // Fallback: if CDN scripts were still loading, retry on window.load.
    if (!window.renderMathInElement) {
      window.addEventListener('load', function () { renderKaTeX(el); });
    }
  });

}());
