(function () {
  var nav = document.getElementById('tutorial-nav');
  if (!nav) return;

  var active = nav.getAttribute('data-active') || 'index.html';
  var tutorials = [
    { href: 'structural_svk.html', label: 'From a mesh to a ROM' },
    { href: 'karman.html', label: 'Kármán vortex street' },
    { href: 'full_order_model.html', label: 'Building a full-order model' },
    { href: 'symbolics_ext.html', label: 'Symbolic definition of a full-order model' },
    { href: 'multiindex_sets.html', label: 'Monomials and multiindices' }
  ];

  function activeClass(href) {
    return href === active ? ' class="active"' : '';
  }

  nav.innerHTML =
    '<a class="docs-install-btn' + (active === 'installation.html' ? ' active' : '') + '" href="installation.html">Installation guide<span>→</span></a>' +
    '<ul class="tutorial-utility-links"><li><a href="../documentation.html">Code documentation</a></li></ul>' +
    '<div class="ver tutorial-nav-heading"><span><a href="index.html"' + activeClass('index.html') + '>Tutorials</a></span></div>' +
    '<ul class="tutorial-list">' + tutorials.map(function (tutorial) {
      return '<li><a href="' + tutorial.href + '"' + activeClass(tutorial.href) + '>' + tutorial.label + '</a></li>';
    }).join('') + '</ul>';
})();
