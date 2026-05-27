var TeamData = (function () {

  var developers =
    '<div class="team-head">' +
    '  <div>' +
    '    <div class="num">Group · 01</div>' +
    '    <h2>Developers.</h2>' +
    '  </div>' +
    '  <p class="desc">Who writes the Julia code — the people behind the commits on' +
    '    <a href="https://github.com/MORFEproject/MORFE.jl" style="color:var(--accent);">MORFEproject/MORFE.jl</a>.' +
    '  </p>' +
    '</div>' +
    '<div class="team-grid two-col">' +

    '  <div class="member">' +
    '    <div class="portrait purple"><img src="assets/portraits/tiago-martins.jpg" alt="Tiago Martins" style="width:100%;height:100%;object-fit:cover;border-radius:3px;display:block;" /></div>' +
    '    <h3>Tiago Martins</h3>' +
    '    <p class="role"><span class="accent">Developer</span> · MORFE.jl</p>' +
    '    <p class="affil">Chair of Applied Mechanics · TU Munich</p>' +
    '    <p class="bio">Cohomological equations, multiindices, and polynomial algebra</p>' +
    '    <div class="links">' +
    '      <a href="https://orcid.org/0000-0002-3200-2225" target="_blank" rel="noopener">ORCID</a>' +
    '      <a href="https://scholar.google.com/citations?user=W-J6NMAAAAAJ" target="_blank" rel="noopener">Scholar</a>' +
    '      <a href="https://github.com/tiagomrns" target="_blank" rel="noopener">GitHub</a>' +
    '    </div>' +
    '  </div>' +

    '  <div class="member">' +
    '    <div class="portrait blue"><img src="assets/portraits/florian-tuschner.jpg" alt="Florian Tuschner" style="width:100%;height:100%;object-fit:cover;border-radius:3px;display:block;" /></div>' +
    '    <h3>Florian Tuschner</h3>' +
    '    <p class="role"><span style="color:var(--jl-blue)">Developer</span> · MORFE.jl</p>' +
    '    <p class="affil">TU Bergakademie Freiberg</p>' +
    '    <p class="bio">Strong form PDEs, full order models, and eigenproblems</p>' +
    '    <div class="links">' +
    '      <a href="https://www.linkedin.com/in/florian-tuschner-1105251b5/" target="_blank" rel="noopener">LinkedIn</a>' +
    '      <a href="https://github.com/FlorianTuschner" target="_blank" rel="noopener">GitHub</a>' +
    '    </div>' +
    '  </div>' +

    '</div>';

  var projectTeam =
    '<div class="team-head">' +
    '  <div>' +
    '    <div class="num">Group · 02</div>' +
    '    <h2>Project team.</h2>' +
    '  </div>' +
    '  <p class="desc">Drives the scientific direction of MORFE.jl.</p>' +
    '</div>' +
    '<div class="team-grid">' +

    '  <div class="member">' +
    '    <div class="portrait purple"><img src="assets/portraits/alessandra-vizzaccaro.jpg" alt="Alessandra Vizzaccaro" style="width:100%;height:100%;object-fit:cover;border-radius:3px;display:block;" /></div>' +
    '    <h3>Alessandra Vizzaccaro</h3>' +
    '    <p class="role"><span class="accent">Scientific lead</span> · Method · MORFE creator</p>' +
    '    <p class="affil">Politecnico di Milano</p>' +
    '    <p class="bio">Drove the FE-native DPIM formulation. One of the original architects of the MORFE codebase.</p>' +
    '    <div class="links">' +
    '      <a href="https://orcid.org/0000-0002-2040-4753" target="_blank" rel="noopener">ORCID</a>' +
    '      <a href="https://scholar.google.com/citations?user=6b4dn8MAAAAJ" target="_blank" rel="noopener">Scholar</a>' +
    '      <a href="https://github.com/av3116" target="_blank" rel="noopener">GitHub</a>' +
    '    </div>' +
    '  </div>' +

    '  <div class="member">' +
    '    <div class="portrait green"><img src="assets/portraits/ulrich-roemer.jpg" alt="Ulrich Römer" style="width:100%;height:100%;object-fit:cover;border-radius:3px;display:block;" /></div>' +
    '    <h3>Ulrich Römer</h3>' +
    '    <p class="role"><span style="color: var(--jl-green);">PI</span> · Julia · Supervision</p>' +
    '    <p class="affil">IMFD · TU Bergakademie Freiberg</p>' +
    '    <p class="bio">Model order reduction, invariant manifolds, and data-driven methods for high-dimensional FE models.</p>' +
    '    <div class="links">' +
    '      <a href="https://tu-freiberg.de/en/facult4/imfd/our-team/prof-dr-ing-u-roemer" target="_blank" rel="noopener">Profile ↗</a>' +
    '      <a href="https://scholar.google.com/citations?user=SHFgnksAAAAJ" target="_blank" rel="noopener">Scholar</a>' +
    '      <a href="https://orcid.org/0000-0002-6393-6063" target="_blank" rel="noopener">ORCID</a>' +
    '    </div>' +
    '  </div>' +

    '  <div class="member">' +
    '    <div class="portrait red"><img src="assets/portraits/francesco-trainotti.jpg" alt="Francesco Trainotti" style="width:100%;height:100%;object-fit:cover;border-radius:3px;display:block;" /></div>' +
    '    <h3>Francesco Trainotti</h3>' +
    '    <p class="role"><span style="color: var(--jl-red);">Industrial Lead</span> · Applications</p>' +
    '    <p class="affil">Chair of Applied Mechanics · TU Munich</p>' +
    '    <p class="bio">Nonlinear vibration analysis, frequency-based substructuring, and dynamic system identification.</p>' +
    '    <div class="links">' +
    '      <a href="https://scholar.google.com/citations?user=CIxMazEAAAAJ" target="_blank" rel="noopener">Scholar</a>' +
    '      <a href="https://orcid.org/0000-0003-4803-2821" target="_blank" rel="noopener">ORCID</a>' +
    '    </div>' +
    '  </div>' +

    '</div>';

  return { developers: developers, projectTeam: projectTeam };

})();
