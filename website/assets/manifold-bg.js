/* MORFE.jl — animated manifold backdrop
   Renders a slowly-drifting wireframe surface (a parameterised invariant manifold)
   responsive to scroll & pointer. Pure 2D canvas with 3D math projected.
*/
(function () {
  const canvas = document.getElementById('manifold-bg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W = 0, H = 0, DPR = Math.min(window.devicePixelRatio || 1, 2);
  function resize() {
    W = canvas.clientWidth = window.innerWidth;
    H = canvas.clientHeight = window.innerHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  // Pointer & scroll state
  let mx = 0.5, my = 0.4; // normalized 0..1
  let tmx = 0.5, tmy = 0.4;
  let scrollY = 0;
  window.addEventListener('pointermove', (e) => {
    tmx = e.clientX / window.innerWidth;
    tmy = e.clientY / window.innerHeight;
  }, { passive: true });
  window.addEventListener('scroll', () => {
    scrollY = window.scrollY;
  }, { passive: true });

  // Surface grid
  const NU = 38, NV = 22;
  const us = [], vs = [];
  for (let i = 0; i < NU; i++) us.push(i / (NU - 1));
  for (let j = 0; j < NV; j++) vs.push(j / (NV - 1));

  // Project a 3D point (x,y,z) to 2D, given camera angles
  function project(x, y, z, cx, cy, cz, sx, sy, sz, scale, ox, oy) {
    // Rotate around Y by ay, then around X by ax
    let X = x, Y = y, Z = z;
    // Yaw (around Y)
    let X1 = X * cy + Z * sy;
    let Z1 = -X * sy + Z * cy;
    // Pitch (around X)
    let Y2 = Y * cx - Z1 * sx;
    let Z2 = Y * sx + Z1 * cx;
    // simple perspective
    const persp = 1 / (1 + Z2 * 0.18);
    return [ox + X1 * scale * persp, oy + Y2 * scale * persp, Z2];
  }

  // The manifold height function — a curved saddle-ish surface, gently animated.
  // Mimics a 2D invariant manifold parameterised by (u,v) -> (x,y,z(x,y,t))
  function surfPoint(u, v, t, warp) {
    const x = (u - 0.5) * 2;
    const y = (v - 0.5) * 2;
    // dominant mode
    let z = 0.55 * Math.sin(2.4 * x + 0.6 * t) * Math.cos(1.8 * y - 0.4 * t);
    // gentle saddle to give the manifold curvature
    z += 0.18 * (x * x - y * y);
    // pointer warp — a soft bump that follows the cursor
    const dx = x - (warp.x);
    const dy = y - (warp.y);
    const d2 = dx * dx + dy * dy;
    z += warp.amp * Math.exp(-d2 * 1.4);
    return [x, y, z];
  }

  // Trajectory on the manifold — a slow oscillation with shrinking amplitude
  // (visualises a flow on the invariant manifold)
  const trail = [];
  const TRAIL_LEN = 220;

  let t0 = performance.now();
  // No roll — keep these constants in scope for project()
  const cz_ = 1, sz_ = 0;

  function frame(now) {
    const t = (now - t0) / 1000;

    // smooth pointer & scroll
    mx += (tmx - mx) * 0.04;
    my += (tmy - my) * 0.04;

    const scrollNorm = Math.min(scrollY / 1800, 1.2);

    // Camera angles — yaw drifts, pitch driven by pointer + scroll
    const ay = t * 0.07 + (mx - 0.5) * 0.6;
    const ax = 0.55 + (my - 0.5) * 0.35 + scrollNorm * 0.25;
    const cx = Math.cos(ax), sx = Math.sin(ax);
    const cy = Math.cos(ay), sy = Math.sin(ay);

    // Pointer warp on surface (in world coords roughly -1..1)
    const warp = {
      x: (mx - 0.5) * 1.6,
      y: (0.5 - my) * 1.0,
      amp: 0.35,
    };

    const ox = W * 0.5;
    const oy = H * 0.55 - scrollNorm * 80;
    const baseScale = Math.min(W, H) * 0.42;

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Compute grid points
    const pts = new Array(NU);
    for (let i = 0; i < NU; i++) {
      pts[i] = new Array(NV);
      for (let j = 0; j < NV; j++) {
        const [x, y, z] = surfPoint(us[i], vs[j], t * 0.5, warp);
        pts[i][j] = project(x, y, z, cx, cy, cz_, sx, sy, sz_, baseScale, ox, oy);
      }
    }

    // U-lines (constant v)
    ctx.lineWidth = 1;
    for (let j = 0; j < NV; j++) {
      ctx.beginPath();
      for (let i = 0; i < NU; i++) {
        const p = pts[i][j];
        // depth-based alpha
        const depth = (p[2] + 1.2) / 2.4;
        const a = 0.05 + 0.22 * Math.max(0, Math.min(1, depth));
        if (i === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
        // we'll stroke the whole line at once; modulate via gradient stroke later
        ctx.strokeStyle = `rgba(149,88,178,${a})`;
      }
      ctx.stroke();
    }

    // V-lines (constant u)
    for (let i = 0; i < NU; i++) {
      ctx.beginPath();
      for (let j = 0; j < NV; j++) {
        const p = pts[i][j];
        if (j === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
        const depth = (p[2] + 1.2) / 2.4;
        const a = 0.04 + 0.18 * Math.max(0, Math.min(1, depth));
        ctx.strokeStyle = `rgba(64,99,216,${a})`;
      }
      ctx.stroke();
    }

    // Trajectory: a shrinking oscillation along (u(t), v(t)) projecting onto the manifold
    const rho = 0.55 * Math.exp(-((t * 0.05) % 6) * 0.0); // keep alive
    const uu = 0.5 + 0.42 * Math.cos(t * 1.2) * (0.6 + 0.4 * Math.sin(t * 0.18));
    const vv = 0.5 + 0.42 * Math.sin(t * 1.5) * (0.6 + 0.4 * Math.cos(t * 0.21));
    const [tx, ty, tz] = surfPoint(uu, vv, t * 0.5, warp);
    const tp = project(tx, ty, tz + 0.02, cx, cy, cz_, sx, sy, sz_, baseScale, ox, oy);
    trail.push(tp);
    if (trail.length > TRAIL_LEN) trail.shift();

    // draw trail
    for (let k = 1; k < trail.length; k++) {
      const a = (k / trail.length);
      ctx.strokeStyle = `rgba(203,60,51,${0.05 + a * 0.65})`;
      ctx.lineWidth = 0.6 + a * 1.4;
      ctx.beginPath();
      ctx.moveTo(trail[k - 1][0], trail[k - 1][1]);
      ctx.lineTo(trail[k][0], trail[k][1]);
      ctx.stroke();
    }
    // head dot
    if (trail.length) {
      const head = trail[trail.length - 1];
      ctx.fillStyle = '#cb3c33';
      ctx.beginPath();
      ctx.arc(head[0], head[1], 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(203,60,51,0.18)';
      ctx.beginPath();
      ctx.arc(head[0], head[1], 10, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
})();
