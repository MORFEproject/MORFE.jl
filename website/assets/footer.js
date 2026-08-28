/* MORFE.jl — shared site footer.
   Set data-base="../" when mounting from a nested directory and optionally set
   data-repo / data-package for a companion-package page. */
(function () {
  var footer = document.getElementById('site-footer');
  if (!footer) return;

  var base = footer.getAttribute('data-base') || '';
  var repo = footer.getAttribute('data-repo') || 'https://github.com/MORFEproject/MORFE.jl';
  var packageName = footer.getAttribute('data-package') || 'MORFE.jl';
  var description = footer.getAttribute('data-description') ||
    'Model-Order Reduction for Finite Elements — direct parametrisation of invariant manifolds in Julia.';

  footer.innerHTML =
    '<div class="wrap">' +
      '<div class="foot-grid">' +
        '<div>' +
          '<a class="nav-logo" href="' + base + 'index.html" style="margin-bottom:14px;">' +
            '<span class="dot"></span> MORFE<span style="color:var(--ink-3)">.jl</span>' +
          '</a>' +
          '<p style="color:var(--ink-3);font-size:13px;max-width:36ch;">' + description + '</p>' +
        '</div>' +
        '<div><h4>Project</h4><ul>' +
          '<li><a href="' + base + 'features.html">Features</a></li>' +
          '<li><a href="' + base + 'tutorials/">Tutorials</a></li>' +
          '<li><a href="' + base + 'documentation.html">Code Documentation</a></li>' +
        '</ul></div>' +
        '<div><h4>Community</h4><ul>' +
          '<li><a href="' + base + 'team.html">Team</a></li>' +
          '<li><a href="' + base + 'publications.html">Publications</a></li>' +
        '</ul></div>' +
        '<div><h4>Code</h4><ul>' +
          '<li><a href="' + repo + '" target="_blank" rel="noopener">GitHub</a></li>' +
          '<li><a href="' + repo + '/releases" target="_blank" rel="noopener">Releases</a></li>' +
          '<li><a href="' + repo + '/issues" target="_blank" rel="noopener">Issue tracker</a></li>' +
          '<li><a href="' + repo + '/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">Contributing</a></li>' +
        '</ul></div>' +
      '</div>' +
      '<div class="foot-meta">' +
        '<div>© MORFE contributors · MIT license</div>' +
        '<div class="mono">' + packageName + ' · Julia</div>' +
      '</div>' +
    '</div>';
})();
