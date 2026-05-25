(function () {
  // Documenter places assets in <head>, so document.body is null when this
  // script first runs. Defer actual injection to DOMContentLoaded.
  function inject() {
    var cssLink = document.querySelector('link[href*="assets/custom.css"]');
    if (!cssLink) return;
    var docsRoot = cssLink.getAttribute('href').replace('assets/custom.css', '');
    var siteRoot = docsRoot + '../';

    var nav = document.createElement('div');
    nav.className = 'site-nav-injected';
    nav.innerHTML =
      '<div class="nav-inner">' +
        '<a href="' + siteRoot + 'index.html" class="nav-logo">' +
          '<span class="dot"></span>' +
          ' MORFE<span style="color:var(--ink-3)">.jl</span>' +
        '</a>' +
        '<div class="nav-links">' +
          '<a href="' + siteRoot + 'features.html">Features</a>' +
          '<a href="' + siteRoot + 'gallery.html">Gallery</a>' +
          '<a href="' + siteRoot + 'tutorials.html">Tutorials</a>' +
          '<a href="' + siteRoot + 'publications.html">Publications</a>' +
          '<a href="' + siteRoot + 'team.html">Team</a>' +
          '<a href="' + siteRoot + 'api.html" class="active">API</a>' +
        '</div>' +
        '<a class="nav-cta" href="https://github.com/MORFEproject/MORFE.jl" target="_blank" rel="noopener">GitHub</a>' +
      '</div>';

    document.body.insertBefore(nav, document.body.firstChild);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }
})();
