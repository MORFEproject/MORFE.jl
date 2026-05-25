// dpim-scheme.jsx
// Interactive 3D DPIM schematic. Mounted via window.MountDpimScheme(rootElOrId).
//
// Depends on globals: React, ReactDOM, katex.
// Equations and view constants are at the top, edit them freely.

(function () {
  const { useState, useEffect, useRef, useMemo, useCallback, useLayoutEffect } = React;

  /* ═══════════════ EDIT DPIM EQUATIONS HERE ══════════════════════════
     Ψ : (q₁, q₂, t) → (y₁, y₂, y₃)                                     */
  function mapToY(q1, q2, t) {
    return [
      q1, // + q2*q2*Math.sin(t),
      q2 + 0.01 * q1 * q1,
      0.3 * q1 + 0.2 * Math.cos(t)
        + 0.01 * Math.sin(t) * (q1 * q1 * q1 + q2 * q2 * q2 * q2),
    ];
  }

  // Reduced dynamics on q-space
  const R_SS = 2.0;
  const B    = 2.0;
  const W    = 1.0;
  const LAM  = 0.1;

  function qOfT(t, r0, delta /*, epsilon */) {
    const r_t = R_SS - (R_SS - r0) * Math.exp(-LAM * t);
    const q1 = 0.6 * r_t * Math.cos(B * t + delta);
    const q2 = 0.8 * r_t * Math.sin(B * t + delta + 0.5);
    return [q1, q2, r_t];
  }

  const U_MIN = -2.8, U_MAX = 2.8;
  const V_MIN = -2.8, V_MAX = 2.8;

  // Scene / view
  const Q_SCALE   = 50;
  const Y_SCALE   = 70;
  const ROT_SPEED = 0.10;
  const TRAIL_LEN = 300;
  const MESH_N    = 12;
  /* ══════════════════════════════════════════════════════════════════ */

  // Orthonormal 3D camera (yaw around z, then pitch around x)
  function rotate3D(p, yaw, pitch) {
    const [x, y, z] = p;
    const cY = Math.cos(yaw),  sY = Math.sin(yaw);
    const cP = Math.cos(pitch), sP = Math.sin(pitch);
    const xr =  x * cY - y * sY;
    const yr =  x * sY + y * cY;
    const yp =  yr * cP - z * sP;
    const zp =  yr * sP + z * cP;
    return [xr, yp, zp];
  }
  const project  = (p, yaw, pitch) => { const r = rotate3D(p, yaw, pitch); return [r[0], -r[2]]; };
  const depthOf  = (p, yaw, pitch) => rotate3D(p, yaw, pitch)[1];

  // Decompose an initial (q₁₀, q₂₀) into (r₀, δ) for the reduced ODE
  function computeParams(q10, q20) {
    const A = Math.cos(0.5), B_ = Math.sin(0.5);
    const x = q10 / 0.6;
    const y = ((q20 / 0.8) - x * B_) / A;
    const r0 = Math.sqrt(x * x + y * y);
    if (r0 < 1e-6) return { r0: 0.001, delta: 0, epsilon: 0 };
    const delta = Math.atan2(3 * q20 - 4 * q10 * B_, 4 * q10 * A);
    return { r0, delta, epsilon: 0 };
  }

  const ACCENT = '#9558b2';
  const MANIFOLD_OPACITY = 0.16;

  // KaTeX-in-SVG label
  function TeX({ x, y, tex, anchor='start', color='#e8e8ee', size=13, italic=false }) {
    const ref = useRef(null);
    useLayoutEffect(() => {
      if (ref.current && window.katex) {
        try { window.katex.render(tex, ref.current, { throwOnError:false, displayMode:false }); }
        catch (e) { ref.current.textContent = tex; }
      }
    }, [tex]);
    const tx = anchor === 'middle' ? 'translate(-50%, -50%)'
             : anchor === 'end'    ? 'translate(-100%, -50%)'
             :                       'translate(0, -50%)';
    return (
      <foreignObject x={x} y={y} width="1" height="1" style={{ overflow: 'visible' }}>
        <div xmlns="http://www.w3.org/1999/xhtml"
          className="dpim-tex"
          style={{
            position: 'absolute', transform: tx,
            color, fontSize: size + 'px',
            fontStyle: italic ? 'italic' : 'normal',
          }}>
          <span ref={ref}/>
        </div>
      </foreignObject>
    );
  }

  function DpimScheme() {
    const [animT, setAnimT] = useState(0);
    const [initQ1, setInitQ1] = useState(1.5);
    const [initQ2, setInitQ2] = useState(1.8);
    const [isDragging, setIsDragging] = useState(false);
    const [dragQ1, setDragQ1] = useState(1.5);
    const [dragQ2, setDragQ2] = useState(1.8);

    const [yaw,   setYaw]   = useState(-0.55);
    const [pitch, setPitch] = useState(0.45);
    const [isRotating, setIsRotating] = useState(false);
    const rotStartRef = useRef({ x: 0, y: 0, yaw: 0, pitch: 0 });

    const svgRef = useRef(null);
    const rafRef = useRef(null);
    const wallClockStartRef = useRef(performance.now());
    const trailQRef = useRef([]);
    const trailYRef = useRef([]);
    const lastSampleRef = useRef(0);

    const effective = isDragging ? { q1: dragQ1, q2: dragQ2 } : { q1: initQ1, q2: initQ2 };
    const { r0, delta, epsilon } = useMemo(
      () => computeParams(effective.q1, effective.q2),
      [effective.q1, effective.q2]
    );

    const [q1, q2, r_t] = useMemo(
      () => qOfT(animT, r0, delta, epsilon),
      [animT, r0, delta, epsilon]
    );
    const [y1, y2, y3] = mapToY(q1, q2, animT);

    const qToSvg = useCallback((u, v) => [110 + u * Q_SCALE, 110 - v * Q_SCALE], []);
    const svgToQ = useCallback((sx, sy) => [(sx - 110) / Q_SCALE, (110 - sy) / Q_SCALE], []);
    const clampQ = (a, b) => [Math.max(-2.1, Math.min(2.1, a)), Math.max(-2.1, Math.min(2.1, b))];

    // animation + auto-rotate
    useEffect(() => {
      let lastNow = performance.now();
      const loop = (now) => {
        const dt = (now - lastNow) / 1000;
        lastNow = now;
        if (!isDragging) setAnimT((now - wallClockStartRef.current) / 1000);
        if (!isRotating) setYaw((y) => y + ROT_SPEED * dt);
        rafRef.current = requestAnimationFrame(loop);
      };
      rafRef.current = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(rafRef.current);
    }, [isDragging, isRotating]);

    // trail
    useEffect(() => {
      if (isDragging) { trailQRef.current = []; trailYRef.current = []; lastSampleRef.current = 0; return; }
      if (animT - lastSampleRef.current < 0.025 && trailQRef.current.length > 0) return;
      lastSampleRef.current = animT;
      trailQRef.current = [...trailQRef.current.slice(-(TRAIL_LEN - 1)), qToSvg(q1, q2)];
      trailYRef.current = [...trailYRef.current.slice(-(TRAIL_LEN - 1)), [y1, y2, y3]];
    }, [animT, q1, q2, y1, y2, y3, isDragging, qToSvg]);

    useEffect(() => {
      trailQRef.current = []; trailYRef.current = []; lastSampleRef.current = 0;
    }, [initQ1, initQ2]);

    const getPosInSvg = useCallback((e) => {
      const rect = svgRef.current.getBoundingClientRect();
      const vw = 1040, vh = 440;
      const cx = e.touches ? e.touches[0].clientX : e.clientX;
      const cy = e.touches ? e.touches[0].clientY : e.clientY;
      return [(cx - rect.left) * vw / rect.width, (cy - rect.top) * vh / rect.height];
    }, []);

    const qBox = { x: 40,  y: 90, w: 220, h: 220 };
    const yBox = { x: 430, y: 40, w: 580, h: 360 };

    const handleDown = useCallback((e) => {
      const [sx, sy] = getPosInSvg(e);
      if (sx >= yBox.x && sx <= yBox.x + yBox.w && sy >= yBox.y && sy <= yBox.y + yBox.h) {
        e.preventDefault();
        setIsRotating(true);
        rotStartRef.current = { x: sx, y: sy, yaw, pitch };
        return;
      }
      if (sx >= qBox.x && sx <= qBox.x + qBox.w && sy >= qBox.y && sy <= qBox.y + qBox.h) {
        e.preventDefault();
        const [a, b] = clampQ(...svgToQ(sx - qBox.x, sy - qBox.y));
        setDragQ1(a); setDragQ2(b);
        setInitQ1(a); setInitQ2(b);
        setIsDragging(true);
        setAnimT(0); wallClockStartRef.current = performance.now();
        trailQRef.current = []; trailYRef.current = []; lastSampleRef.current = 0;
      }
    }, [getPosInSvg, svgToQ, yaw, pitch]);

    const handleMove = useCallback((e) => {
      if (isRotating) {
        e.preventDefault();
        const [sx, sy] = getPosInSvg(e);
        const dx = sx - rotStartRef.current.x;
        const dy = sy - rotStartRef.current.y;
        setYaw(rotStartRef.current.yaw + dx * 0.012);
        setPitch(Math.max(-1.4, Math.min(1.4, rotStartRef.current.pitch - dy * 0.012)));
        return;
      }
      if (!isDragging) return;
      e.preventDefault();
      const [sx, sy] = getPosInSvg(e);
      const bx = Math.max(0, Math.min(qBox.w, sx - qBox.x));
      const by = Math.max(0, Math.min(qBox.h, sy - qBox.y));
      const [a, b] = clampQ(...svgToQ(bx, by));
      setDragQ1(a); setDragQ2(b);
      setInitQ1(a); setInitQ2(b);
    }, [isDragging, isRotating, getPosInSvg, svgToQ]);

    const handleUp = useCallback(() => {
      if (isDragging) {
        setIsDragging(false);
        setAnimT(0); wallClockStartRef.current = performance.now();
        trailQRef.current = []; trailYRef.current = []; lastSampleRef.current = 0;
      }
      if (isRotating) setIsRotating(false);
    }, [isDragging, isRotating]);

    useEffect(() => {
      if (!isDragging && !isRotating) return;
      const m = (e) => handleMove(e);
      const u = () => handleUp();
      window.addEventListener('mousemove', m);
      window.addEventListener('mouseup',   u);
      window.addEventListener('touchmove', m, { passive: false });
      window.addEventListener('touchend',  u);
      return () => {
        window.removeEventListener('mousemove', m);
        window.removeEventListener('mouseup',   u);
        window.removeEventListener('touchmove', m);
        window.removeEventListener('touchend',  u);
      };
    }, [isDragging, isRotating, handleMove, handleUp]);

    const proj  = useCallback((p3) => project([p3[0] * Y_SCALE, p3[1] * Y_SCALE, p3[2] * Y_SCALE], yaw, pitch), [yaw, pitch]);
    const projU = useCallback((p3) => project(p3, yaw, pitch), [yaw, pitch]);
    const depth = useCallback((p3) => depthOf([p3[0] * Y_SCALE, p3[1] * Y_SCALE, p3[2] * Y_SCALE], yaw, pitch), [yaw, pitch]);

    const meshQuads = useMemo(() => {
      const N = MESH_N, quads = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const u0 = U_MIN + (U_MAX - U_MIN) * i / N;
        const u1 = U_MIN + (U_MAX - U_MIN) * (i + 1) / N;
        const v0 = V_MIN + (V_MAX - V_MIN) * j / N;
        const v1 = V_MIN + (V_MAX - V_MIN) * (j + 1) / N;
        const c1 = mapToY(u0, v0, animT);
        const c2 = mapToY(u1, v0, animT);
        const c3 = mapToY(u1, v1, animT);
        const c4 = mapToY(u0, v1, animT);
        const p1 = proj(c1), p2 = proj(c2), p3 = proj(c3), p4 = proj(c4);
        const d = (depth(c1) + depth(c2) + depth(c3) + depth(c4)) / 4;
        quads.push({
          pts: `${p1[0].toFixed(2)},${p1[1].toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)} ${p3[0].toFixed(2)},${p3[1].toFixed(2)} ${p4[0].toFixed(2)},${p4[1].toFixed(2)}`,
          d
        });
      }
      quads.sort((a, b) => b.d - a.d);
      return quads;
    }, [animT, proj, depth]);

    const ridges = useMemo(() => {
      const vs = [V_MIN, 0.0, 0.6, 1.4, V_MAX];
      const samples = 48;
      return vs.map(v => {
        const pts = [];
        for (let i = 0; i <= samples; i++) {
          const u = U_MIN + (U_MAX - U_MIN) * i / samples;
          pts.push(proj(mapToY(u, v, animT)));
        }
        return pts.map(p => p.join(',')).join(' ');
      });
    }, [animT, proj]);

    const eigenPoly = useMemo(() => {
      const w = 70, d = 70;
      return [[-w,-d,-0.3*w], [w,-d,0.3*w], [w,d,0.3*w], [-w,d,-0.3*w]]
        .map(c => projU(c).map(n => n.toFixed(2)).join(','))
        .join(' ');
    }, [projU]);

    const axesLen = 70;
    const axEnds = [0, 1, 2].map(i => {
      const v = [0, 0, 0]; v[i] = axesLen;
      return projU(v);
    });

    const [qSvgX, qSvgY] = qToSvg(q1, q2);
    const [iSvgX, iSvgY] = qToSvg(effective.q1, effective.q2);
    const yProj = proj([y1, y2, y3]);
    const qTrailStr = trailQRef.current.map(p => p.join(',')).join(' ');
    const yTrailStr = trailYRef.current.map(p3 => proj(p3).join(',')).join(' ');
    const ringColor = isDragging ? '#e8c84a' : '#389826';
    const envR = 0.8771 * R_SS * Q_SCALE;
    const yC = { cx: yBox.x + yBox.w / 2, cy: yBox.y + yBox.h / 2 + 30 };

    return (
      <svg ref={svgRef} viewBox="0 0 1040 440" role="img"
        style={{ fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" }}
        onMouseDown={handleDown} onTouchStart={handleDown}>
        <defs>
          <marker id="d-arr-ink" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
            <path d="M0,0 L9,4.5 L0,9 z" fill="#b8b8c4"/>
          </marker>
          <marker id="d-arr-acc" markerWidth="8" markerHeight="8" refX="7.5" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={ACCENT}/>
          </marker>
          <pattern id="d-grid" width="22" height="22" patternUnits="userSpaceOnUse">
            <path d="M22 0 H0 V22" stroke="#1f1f2c" strokeWidth="0.6" fill="none"/>
          </pattern>
          <filter id="d-glow">
            <feGaussianBlur stdDeviation="1.8" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <g stroke="#2a2a3a" strokeWidth="1" fill="none">
          <path d="M16 16 H40 M16 16 V40"/>
          <path d="M1024 16 H1000 M1024 16 V40"/>
          <path d="M16 424 H40 M16 424 V400"/>
          <path d="M1024 424 H1000 M1024 424 V400"/>
        </g>

        {/* ───────── q-space ───────── */}
        <g transform={`translate(${qBox.x},${qBox.y})`}>
          <text x="0" y="-20" fontSize="10" letterSpacing="1.5" fill="#6e6e7e">REDUCED · MASTER COORDS</text>
          <TeX x={5} y={210} tex="\mathbb{R}^d" color="#6e6e7e" size={11}/>

          <rect x="0" y="0" width="220" height="220" fill="url(#d-grid)"
            stroke="#2a2a3a" strokeWidth="1"
            className={'dpim-draggable' + (isDragging ? ' dragging' : '')}/>

          <line x1="5" y1="110" x2="215" y2="110" stroke="#b8b8c4" strokeWidth="1" markerEnd="url(#d-arr-ink)"/>
          <line x1="110" y1="215" x2="110" y2="5" stroke="#b8b8c4" strokeWidth="1" markerEnd="url(#d-arr-ink)"/>
          <TeX x={207} y={122} tex="z_1" color="#e8e8ee" size={14}/>
          <TeX x={90} y={6}   tex="z_2" color="#e8e8ee" size={14}/>
          <circle cx="110" cy="110" r="2.2" fill="#b8b8c4"/>
          <text x="96" y="126" fontSize="10" fill="#6e6e7e">0</text>

          <circle cx="110" cy="110" r={envR}
            stroke="#389826" strokeWidth="0.5" strokeDasharray="3 5" fill="none" opacity="0.35"/>

          {qTrailStr && <polyline points={qTrailStr} stroke="#389826" strokeWidth="1.2" fill="none"
            strokeLinecap="round" strokeLinejoin="round" opacity="0.6" style={{ pointerEvents: 'none' }}/>}

          <circle cx={iSvgX} cy={iSvgY} r="7" fill="none" stroke={ringColor}
            strokeWidth={isDragging ? 2.4 : 1.8} strokeOpacity={isDragging ? 1 : 0.75}
            filter={isDragging ? 'url(#d-glow)' : 'none'} style={{ pointerEvents: 'none' }}/>
          <circle cx={iSvgX} cy={iSvgY} r="2" fill={ringColor} style={{ pointerEvents: 'none' }}/>

          {!isDragging && (
            <g style={{ pointerEvents: 'none' }}>
              <circle cx={qSvgX} cy={qSvgY} r="3.5" fill="#389826" filter="url(#d-glow)"/>
              <circle cx={qSvgX} cy={qSvgY} r="7"   fill="none" stroke="#389826" strokeWidth="0.8" strokeOpacity="0.5"/>
              <TeX x={qSvgX + 8} y={qSvgY - 5} tex="q(t)" color="#389826" size={12} italic/>
            </g>
          )}

          {/*<TeX x={0} y={240} tex="\dot{q} = f(q)" color="#b8b8c4" size={11}/>*/}
          <text x="0"  y="240" fontSize="10" fill="#6e6e7e" textAnchor="start">t={animT.toFixed(2)}s</text>
          <text x="88" y="240" fontSize="10" fill="#6e6e7e" textAnchor="start">z₁={q1.toFixed(2)}</text>
          <text x="172" y="240" fontSize="10" fill="#6e6e7e" textAnchor="start">z₂={q2.toFixed(2)}</text>
        </g>

        {/* ───────── Ψ arrow ───────── */}
        <g transform="translate(290,220)">
          <line x1="0" y1="19" x2="130" y2="19" stroke="#1f1f2c" strokeWidth="1" strokeDasharray="2 4"/>
          <path d="M 4,2 C 36,-32 96,-32 128,2" stroke={ACCENT} strokeWidth="1.4" fill="none" markerEnd="url(#d-arr-acc)"/>
          <TeX x={66} y={-35} tex="\mathbf{W}:\mathbb{R}^d \to \mathbb{R}^N" color="#e8e8ee" size={13} anchor="middle"/>
          <rect x="38" y="10" width="56" height="18" rx="2" fill={ACCENT + '1f'} stroke={ACCENT} strokeWidth="0.8"/>
          <text x="66" y="22" fontSize="10" letterSpacing="1.6" fill={ACCENT} textAnchor="middle">DPIM</text>
          <text x="66" y="50" fontSize="9" letterSpacing="1.2" fill="#6e6e7e" textAnchor="middle">ORDER 3 · POLYNOMIAL</text>
        </g>

        {/* ───────── y-space ───────── */}
        <g>
          <text x={yBox.x} y={yBox.y - 20} fontSize="10" letterSpacing="1.5" fill="#6e6e7e">FULL · FE COORDS</text>
          <TeX x={yBox.x + 5} y={yBox.y + yBox.h - 10} tex="\mathbb{R}^N" color="#6e6e7e" size={11}/>

          <rect x={yBox.x} y={yBox.y} width={yBox.w} height={yBox.h}
            fill="transparent" stroke="#2a2a3a" strokeWidth="1"
            className={'dpim-rotate' + (isRotating ? ' rotating' : '')}/>

          <g transform={`translate(${yC.cx},${yC.cy})`} style={{ pointerEvents: 'none' }}>
            <polygon points={eigenPoly} fill="#4063d8" fillOpacity="0.06"
              stroke="#4063d8" strokeWidth="0.7" strokeDasharray="3 3"/>

            {meshQuads.map((q, i) => (
              <polygon key={i} points={q.pts}
                fill={ACCENT} fillOpacity={MANIFOLD_OPACITY}
                stroke={ACCENT} strokeWidth="0.35" strokeOpacity="0.5"/>
            ))}

            {ridges.map((pts, i) => (
              <polyline key={i} points={pts} stroke={ACCENT} strokeWidth="0.7"
                strokeOpacity={0.55} fill="none" strokeDasharray="2 3"/>
            ))}

            {axEnds.map((p, i) => (
              <line key={i} x1="0" y1="0" x2={p[0]} y2={p[1]}
                stroke="#b8b8c4" strokeWidth="1" markerEnd="url(#d-arr-ink)"/>
            ))}
            <circle cx="0" cy="0" r="2.4" fill="#b8b8c4"/>

            {yTrailStr && <polyline points={yTrailStr} stroke={ACCENT} strokeWidth="1.2" fill="none"
              strokeLinecap="round" strokeLinejoin="round" opacity="0.6"/>}

            {!isDragging && (
              <g>
                <circle cx={yProj[0]} cy={yProj[1]} r="4.5" fill={ACCENT} filter="url(#d-glow)"/>
                <circle cx={yProj[0]} cy={yProj[1]} r="9"   fill="none" stroke={ACCENT} strokeWidth="0.9" strokeOpacity="0.45"/>
              </g>
            )}
          </g>

          {axEnds.map((p, i) => (
            <TeX key={i}
              x={yC.cx + p[0] + (p[0] > 0 ? 8 : -8)}
              y={yC.cy + p[1] + (p[1] > 0 ? 8 : -8)}
              tex={`u_${i + 1}`} color="#e8e8ee" size={14}
              anchor={p[0] > 0 ? 'start' : 'end'} italic/>
          ))}

          {!isDragging && (
            <TeX x={yC.cx + yProj[0] + 10} y={yC.cy + yProj[1] - 10}
              tex="\mathbf{W}(\mathbf{z}(t))" color={ACCENT} size={12} italic/>
          )}

          <text x={yBox.x + 14} y={yBox.y + 22} fontSize="10" fill="#7a92dc" fontStyle="italic" style={{ pointerEvents: 'none' }}>
            E · master eigenspace
          </text>
          <text x={yBox.x + 14} y={yBox.y + 38} fontSize="11" fill="#c79be0" fontStyle="italic" style={{ pointerEvents: 'none' }}>
            W · invariant manifold
          </text>

          <text x={yBox.x} y={yBox.y + yBox.h + 22} fontSize="9" letterSpacing="1.3" fill="#6e6e7e">
            FULL-ORDER FE MODEL
          </text>
          <text x={yBox.x + yBox.w} y={yBox.y + yBox.h + 22} fontSize="10" fill="#6e6e7e" textAnchor="end">
            u₁={y1.toFixed(2)}  u₂={y2.toFixed(2)}  u₃={y3.toFixed(2)}
          </text>
          <text x={yBox.x + yBox.w} y={yBox.y + yBox.h + 38} fontSize="10" fill="#6e6e7e" textAnchor="end">
            yaw={(yaw * 180 / Math.PI).toFixed(0)}°  pitch={(pitch * 180 / Math.PI).toFixed(0)}°
          </text>
        </g>
      </svg>
    );
  }

  // Public mounting helper. Pass a DOM element or its id.
  window.MountDpimScheme = function (target) {
    const el = typeof target === 'string' ? document.getElementById(target) : target;
    if (!el) return;
    if (el.__dpimMounted) return;
    el.__dpimMounted = true;
    ReactDOM.createRoot(el).render(<DpimScheme />);
  };

  // Auto-mount: anything with id "dpim-root" or [data-dpim-root] is mounted as soon as
  // this Babel-transformed script runs. Safe to call multiple times.
  function autoMount() {
    document.querySelectorAll('#dpim-root, [data-dpim-root]').forEach(el => window.MountDpimScheme(el));
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoMount);
  } else {
    autoMount();
  }
})();
