/*
 * Double spike toolbox -- browser GUI.
 *
 * Everything runs locally: no network access, no dependencies.  The numerics
 * come from ./math.js, which is a line by line port of the Python package.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const DS = window.DS;

  /* ====================================================================== *
   * state
   * ====================================================================== */
  const state = {
    element: 'Fe',
    standard: null,       // Float64Array, user editable
    isoinv: [],           // isotope numbers, exactly four
    spikeType: 'pure',    // 'pure' | 'real'
    spikeA: null,         // index of the first spiking isotope (pure) or rawspike row (real)
    spikeB: null,
    q: 0.5,               // proportion of spikeA in the double spike
    spike: null,          // Float64Array composition
    emod: {
      intensity: 10, deltat: 8, R: 1e11, T: 300,
      measuredType: 'fixed-total', radiogenic: false,
    },
    errorTarget: 'alpha',
    erNum: 0, erDen: 1,   // indices into the isotope list
    resolution: 150,
    threshold: 0.4,
    tab: 'precision',
    results: null,        // last inversion results
  };

  const sys = () => ELEMENTS[state.element];

  /* ====================================================================== *
   * generic helpers
   * ====================================================================== */
  const fmt = (x, d) => {
    if (x === null || x === undefined || !isFinite(x)) return '—';
    if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e6)) return x.toExponential(d === undefined ? 3 : d);
    return x.toFixed(d === undefined ? 4 : d);
  };
  const pct = (x, d) => (x * 100).toFixed(d === undefined ? 2 : d) + '%';
  const el = (tag, attrs, children) => {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k.startsWith('on')) n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return n;
  };

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ====================================================================== *
   * model plumbing
   * ====================================================================== */
  function isRadiogenic(name) {
    return ['Pb', 'Sr', 'Hf', 'Os', 'Nd'].includes(name);
  }

  function resetStandard() {
    state.standard = Float64Array.from(sys().standard);
  }

  function suggestedIsotopes() {
    const s = sys();
    return s.isonum
      .map((v, i) => [v, s.standard[i]])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map((p) => p[0])
      .sort((a, b) => a - b);
  }

  function buildEmod() {
    const s = sys();
    return DS.buildErrorModel(s.isonum.length, {
      intensity: state.emod.intensity,
      deltat: state.emod.deltat,
      R: state.emod.R,
      T: state.emod.T,
      measuredType: state.emod.measuredType,
      radiogenic: state.emod.radiogenic,
    });
  }

  function isoinvIndices() {
    return DS.isotopeIndices(sys(), state.isoinv);
  }

  function errorRatioIndices() {
    if (state.errorTarget === 'alpha') return null;
    return [state.erNum, state.erDen];
  }

  function buildSpike() {
    const s = sys();
    const n = s.isonum.length;
    const comp = new Float64Array(n);
    if (state.spikeType === 'pure') {
      comp[state.spikeA] = state.q;
      comp[state.spikeB] = 1 - state.q;
    } else {
      const a = s.rawspike[state.spikeA], b = s.rawspike[state.spikeB];
      for (let i = 0; i < n; i++) comp[i] = state.q * a[i] + (1 - state.q) * b[i];
    }
    state.spike = DS.normalise(comp);
    return state.spike;
  }

  function prepFor(spike) {
    const s = sys();
    const emod = buildEmod();
    return DS.prepareSystem(s, spike, isoinvIndices(), errorRatioIndices(), emod);
  }

  function currentError(prop, spike) {
    const s = sys();
    const emod = buildEmod();
    const prep = prepFor(spike);
    return DS.errorEstimate(s, prep, emod, prop, spike, 0, 0);
  }

  /* ====================================================================== *
   * canvas plotting
   * ====================================================================== */
  function setupCanvas(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const h = cssHeight;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) { hi = lo + 1; }
    const span = hi - lo;
    const raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    const start = Math.ceil(lo / step) * step;
    const out = [];
    for (let v = start; v <= hi + step * 1e-9; v += step) out.push(v);
    out.step = step;
    return out;
  }

  /**
   * A formatter for a whole axis.  Deciding the number of decimals from the tick
   * step keeps every label on an axis in the same style -- otherwise a single
   * axis mixes "0.0000" with "2.5000e-4".
   */
  function axisFormatter(ticks) {
    const step = ticks.step || (ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1);
    if (!isFinite(step) || step <= 0) return (v) => String(v);
    if (step >= 1e5) return (v) => v.toExponential(1);
    const dec = Math.max(0, Math.min(9, Math.ceil(-Math.log10(step * 0.999))));
    return (v) => v.toFixed(dec);
  }

  /** Simple line chart.  series: [{x, y, color, label, dash}] */
  function drawLineChart(canvas, series, opts) {
    const height = opts.height || 250;
    const { ctx, w, h } = setupCanvas(canvas, height);
    const pad = { l: 68, r: 14, t: 14, b: 36 };
    const xs = [], ys = [];
    series.forEach((s) => { s.x.forEach((v) => xs.push(v)); s.y.forEach((v) => { if (isFinite(v)) ys.push(v); }); });
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = 0, y1 = Math.max(...ys);
    if (!isFinite(y1) || y1 <= 0) y1 = 1;
    if (opts.yMax !== undefined) y1 = opts.yMax;
    if (opts.xMin !== undefined) x0 = opts.xMin;
    if (opts.xMax !== undefined) x1 = opts.xMax;
    const PX = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r);
    const PY = (y) => h - pad.b - ((y - y0) / (y1 - y0 || 1)) * (h - pad.t - pad.b);

    // grid + axes
    const yt = niceTicks(y0, y1, 5);
    const xt = niceTicks(x0, x1, 5);
    const fy = axisFormatter(yt);
    const fx = axisFormatter(xt);
    ctx.font = '11px ' + css('--mono');
    ctx.strokeStyle = css('--border-soft');
    ctx.fillStyle = css('--text-faint');
    ctx.lineWidth = 1;
    yt.forEach((v) => {
      const y = PY(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(fy(v), pad.l - 8, y);
    });
    xt.forEach((v) => {
      const x = PX(v);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(fx(v), x, h - pad.b + 6);
    });
    ctx.strokeStyle = css('--border');
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
    ctx.stroke();

    // series
    series.forEach((s) => {
      ctx.strokeStyle = s.color || css('--accent');
      ctx.lineWidth = s.width || 1.8;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (!isFinite(y)) { started = false; continue; }
        const px = PX(s.x[i]), py = PY(y);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // markers (optimum)
    (opts.markers || []).forEach((m) => {
      const px = PX(m.x), py = PY(m.y);
      ctx.fillStyle = m.color || css('--warn');
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = css('--panel'); ctx.lineWidth = 1.5; ctx.stroke();
      if (m.label) {
        ctx.fillStyle = css('--text-dim');
        ctx.font = '11px ' + css('--sans');
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(m.label, px + 8, py - 4);
      }
    });

    // axis labels
    ctx.fillStyle = css('--text-dim');
    ctx.font = '11.5px ' + css('--sans');
    if (opts.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(opts.xLabel, (pad.l + w - pad.r) / 2, h - 4); }
    if (opts.yLabel) {
      ctx.save(); ctx.translate(12, (pad.t + h - pad.b) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    }
    return { PX, PY, pad, w, h };
  }

  /* sequential colormap: deep blue -> teal -> yellow */
  const RAMP = [
    [0.00, 31, 60, 110],
    [0.20, 40, 110, 160],
    [0.40, 60, 165, 160],
    [0.60, 150, 200, 120],
    [0.80, 235, 200, 80],
    [1.00, 250, 240, 150],
  ];
  function rampColor(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < RAMP.length; i++) {
      if (t <= RAMP[i][0]) {
        const a = RAMP[i - 1], b = RAMP[i];
        const f = (t - a[0]) / (b[0] - a[0] || 1);
        return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f];
      }
    }
    const l = RAMP[RAMP.length - 1];
    return [l[1], l[2], l[3]];
  }

  /**
   * Filled 2D error map.  grid is a (ny x nx) array of errors, values above vmax
   * are left transparent so that only the region within the threshold shows.
   */
  function drawHeatMap(canvas, grid, nx, ny, x0, x1, y0, y1, vmin, vmax, opts) {
    const height = opts.height || 360;
    const { ctx, w, h } = setupCanvas(canvas, height);
    const pad = { l: 68, r: 122, t: 14, b: 36 };
    const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;

    const off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    const octx = off.getContext('2d');
    const img = octx.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        // grid row 0 corresponds to the top of the plot
        const v = grid[(ny - 1 - j) * nx + i];
        const p = (j * nx + i) * 4;
        if (!isFinite(v) || v > vmax) { img.data[p + 3] = 0; continue; }
        const c = rampColor((v - vmin) / (vmax - vmin || 1));
        img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 235;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, pad.l, pad.t, iw, ih);

    // axes
    ctx.font = '11px ' + css('--mono'); ctx.fillStyle = css('--text-faint');
    ctx.strokeStyle = css('--border');
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, iw, ih);
    const xt = niceTicks(x0, x1, 5), yt = niceTicks(y0, y1, 5);
    const fx = axisFormatter(xt), fy = axisFormatter(yt);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    xt.forEach((v) => ctx.fillText(fx(v), pad.l + ((v - x0) / (x1 - x0)) * iw, pad.t + ih + 6));
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    yt.forEach((v) => ctx.fillText(fy(v), pad.l - 8, pad.t + ih - ((v - y0) / (y1 - y0)) * ih));

    // optimum marker
    const m = opts.marker;
    if (m && isFinite(m.x) && isFinite(m.y)) {
      const px = pad.l + ((m.x - x0) / (x1 - x0)) * iw;
      const py = pad.t + ih - ((m.y - y0) / (y1 - y0)) * ih;
      ctx.fillStyle = '#ff4d4d';
      ctx.beginPath(); ctx.arc(px, py, 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    // colour bar, with its label to the left of the bar so that nothing is
    // clipped by the right hand edge of the canvas
    const bw = 12, bx = w - pad.r + 30, by = pad.t, bh = ih;
    const grad = ctx.createLinearGradient(0, by + bh, 0, by);
    RAMP.forEach((r) => grad.addColorStop(r[0], 'rgb(' + r[1] + ',' + r[2] + ',' + r[3] + ')'));
    ctx.fillStyle = grad;
    ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = css('--border'); ctx.strokeRect(bx, by, bw, bh);
    ctx.fillStyle = css('--text-faint');
    ctx.font = '10.5px ' + css('--mono');
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(opts.vFormat ? opts.vFormat(vmax) : fmt(vmax, 4), bx + bw + 5, by + 6);
    ctx.fillText(opts.vFormat ? opts.vFormat(vmin) : fmt(vmin, 4), bx + bw + 5, by + bh - 6);
    if (opts.cbarLabel) {
      ctx.save();
      ctx.translate(bx - 9, by + bh / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.font = '11px ' + css('--sans');
      ctx.fillText(opts.cbarLabel, 0, 0);
      ctx.restore();
    }

    ctx.fillStyle = css('--text-dim'); ctx.font = '11.5px ' + css('--sans');
    if (opts.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(opts.xLabel, pad.l + iw / 2, h - 4); }
    if (opts.yLabel) {
      ctx.save(); ctx.translate(12, pad.t + ih / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    }
  }

  /* ====================================================================== *
   * control panel construction
   * ====================================================================== */
  function group(title, open) {
    return el('details', { class: 'group', open: open ? '' : null }, [
      el('summary', null, [title]),
      el('div', { class: 'body' }),
    ]);
  }

  function field(label, node, hint) {
    const kids = [];
    const lab = el('label', null, [label]);
    kids.push(lab, node);
    if (hint) kids.push(el('div', { class: 'hint', html: hint }));
    return el('div', { class: 'field' }, kids);
  }

  function numberInput(value, onInput, step) {
    const inp = el('input', { type: 'number', value: String(value), step: step || 'any' });
    inp.addEventListener('input', () => { const v = parseFloat(inp.value); if (isFinite(v)) onInput(v); });
    return inp;
  }

  function select(options, value, onChange) {
    const s = el('select', null, options.map((o) =>
      el('option', { value: String(o.value), selected: String(o.value) === String(value) ? '' : null }, [o.label])));
    s.addEventListener('change', () => onChange(s.value));
    return s;
  }

  const PANEL = $('controls');

  function buildControls() {
    PANEL.innerHTML = '';

    /* ---- isotope system ---- */
    const sysGroup = group('同位素体系', true);
    const sysBody = sysGroup.querySelector('.body');
    const elemOpts = ELEMENT_ORDER.map((e) => ({
      value: e,
      label: e + '  (' + ELEMENTS[e].isonum.length + ' 个同位素' +
        (ELEMENTS[e].rawspike.length ? ', ' + ELEMENTS[e].rawspike.length + ' 种单稀释剂' : '') + ')',
    }));
    sysBody.appendChild(field('元素', select(elemOpts, state.element, (v) => {
      state.element = v; onElementChange();
    })));

    const table = el('table', { class: 'mini' });
    table.innerHTML = '<thead><tr><th>同位素</th><th>质量</th><th>标样丰度</th></tr></thead>';
    const tbody = el('tbody');
    table.appendChild(tbody);
    for (let i = 0; i < sys().isonum.length; i++) {
      const inp = el('input', { type: 'number', step: 'any', value: String(sys().standard[i]) });
      inp.addEventListener('change', () => {
        const v = parseFloat(inp.value);
        if (isFinite(v) && v >= 0) {
          state.standard[i] = v;
          state.standard = DS.normalise(state.standard);
          buildSpike(); render();
        }
      });
      tbody.appendChild(el('tr', null, [
        el('td', { html: '<sup>' + sys().isonum[i] + '</sup>' + state.element }),
        el('td', { html: '<span style="color:var(--text-faint)">' + sys().mass[i].toFixed(5) + '</span>' }),
        el('td', null, [inp]),
      ]));
    }
    sysBody.appendChild(el('div', { class: 'field' }, [table]));
    sysBody.appendChild(el('div', { class: 'toolbar' }, [
      el('button', { class: 'ghost', onclick: () => { resetStandard(); buildSpike(); buildControls(); render(); } }, ['恢复默认标样']),
    ]));
    PANEL.appendChild(sysGroup);

    /* ---- inversion isotopes ---- */
    const invGroup = group('反演同位素（4 个）', true);
    const invBody = invGroup.querySelector('.body');
    const grid = el('div', { class: 'iso-grid' });
    for (let k = 0; k < 4; k++) {
      const opts = sys().isonum.map((v) => ({ value: v, label: v + state.element }));
      grid.appendChild(field('第 ' + (k + 1) + ' 个', select(opts, state.isoinv[k], (v) => {
        const val = parseInt(v, 10);
        if (state.isoinv.includes(val)) {
          const other = state.isoinv.indexOf(val);
          state.isoinv[other] = state.isoinv[k];
        }
        state.isoinv[k] = val;
        buildSpike(); buildControls(); render();
      })));
    }
    invBody.appendChild(grid);
    invBody.appendChild(el('div', { class: 'toolbar' }, [
      el('button', { class: 'ghost', onclick: () => { state.isoinv = suggestedIsotopes(); buildSpike(); buildControls(); render(); } }, ['选最丰富的 4 个']),
    ]));
    invBody.appendChild(el('div', { class: 'hint', html: '反演只用这 4 个同位素（构成 3 个独立比值）。其余同位素仍用于重建样品组成。' }));
    PANEL.appendChild(invGroup);

    /* ---- double spike ---- */
    const spkGroup = group('双稀释剂组成', true);
    const spkBody = spkGroup.querySelector('.body');
    spkBody.appendChild(field('类型', select([
      { value: 'pure', label: '纯稀释剂（两种单同位素混合）' },
      { value: 'real', label: '实际稀释剂（含杂质，取自数据表）' },
    ].filter((o) => o.value === 'pure' || sys().rawspike.length >= 2), state.spikeType, (v) => {
      state.spikeType = v;
      state.spikeA = 0; state.spikeB = 1;
      buildSpike(); buildControls(); render();
    })));

    if (state.spikeType === 'pure') {
      const optsA = sys().isonum.map((v) => ({ value: v, label: v + state.element }));
      const idxA = sys().isonum[state.spikeA] !== undefined ? sys().isonum[state.spikeA] : sys().isonum[0];
      const idxB = sys().isonum[state.spikeB] !== undefined ? sys().isonum[state.spikeB] : sys().isonum[1];
      const row = el('div', { class: 'iso-grid' }, [
        field('稀释剂 A', select(optsA, idxA, (v) => { state.spikeA = sys().isonum.indexOf(parseInt(v, 10)); buildSpike(); buildControls(); render(); })),
        field('稀释剂 B', select(optsA, idxB, (v) => { state.spikeB = sys().isonum.indexOf(parseInt(v, 10)); buildSpike(); buildControls(); render(); })),
      ]);
      spkBody.appendChild(row);
    } else {
      const names = sys().rawspike.map((r, i) => ({ value: i, label: '单稀释剂 ' + (i + 1) + '（' + r.map((v, k) => v > 0.05 ? sys().isonum[k] + state.element : '').filter(Boolean).join('-') + '）' }));
      const row = el('div', { class: 'iso-grid' }, [
        field('稀释剂 A', select(names, state.spikeA, (v) => { state.spikeA = parseInt(v, 10); buildSpike(); render(); })),
        field('稀释剂 B', select(names, state.spikeB, (v) => { state.spikeB = parseInt(v, 10); buildSpike(); render(); })),
      ]);
      spkBody.appendChild(row);
    }

    const qRange = el('input', { type: 'range', min: '0.001', max: '0.999', step: '0.001', value: String(state.q) });
    const qOut = el('span', { class: 'chip', id: 'qout' }, [pct(state.q, 1)]);
    qRange.addEventListener('input', () => {
      state.q = parseFloat(qRange.value);
      qOut.textContent = pct(state.q, 1);
      buildSpike(); render();
    });
    spkBody.appendChild(field('A 占双稀释剂的比例', el('div', { class: 'row' }, [qRange, el('span', { class: 'fixed' }, [qOut])])));

    const chips = el('div', { class: 'chip-row' });
    sys().isonum.forEach((v, i) => {
      chips.appendChild(el('span', { class: 'chip' + (state.spike[i] > 1e-6 ? ' on' : '') },
        [v + state.element + ' ' + (state.spike[i] * 100).toFixed(2) + '%']));
    });
    spkBody.appendChild(el('div', { class: 'field' }, [chips]));
    PANEL.appendChild(spkGroup);

    /* ---- error model ---- */
    const emGroup = group('误差模型', false);
    const emBody = emGroup.querySelector('.body');
    emBody.appendChild(el('div', { class: 'iso-grid' }, [
      field('总束流强度 (V)', numberInput(state.emod.intensity, (v) => { state.emod.intensity = v; render(); })),
      field('积分时间 (s)', numberInput(state.emod.deltat, (v) => { state.emod.deltat = v; render(); })),
    ]));
    emBody.appendChild(el('div', { class: 'iso-grid' }, [
      field('电阻 (Ω)', numberInput(state.emod.R, (v) => { state.emod.R = v; render(); })),
      field('温度 (K)', numberInput(state.emod.T, (v) => { state.emod.T = v; render(); })),
    ]));
    emBody.appendChild(field('束流归一方式', select([
      { value: 'fixed-total', label: '固定总束流（稀释剂+样品）' },
      { value: 'fixed-sample', label: '固定样品束流' },
    ], state.emod.measuredType, (v) => { state.emod.measuredType = v; render(); })));
    const radChk = el('input', { type: 'checkbox', checked: state.emod.radiogenic ? '' : null });
    radChk.addEventListener('change', () => { state.emod.radiogenic = radChk.checked; render(); });
    emBody.appendChild(el('div', { class: 'row' }, [
      el('label', { class: 'fixed', style: 'font-size:12px;color:var(--text-dim)' }, [radChk, ' 把标样（未稀释）的测量误差计入']),
    ]));
    emBody.appendChild(el('div', { class: 'hint', html: '噪声模型：σ² = a + b·I + c·I²（式 34）。a 为 Johnson–Nyquist 噪声，b 为计数统计。默认对标样不计误差（非放射性体系）。' }));
    PANEL.appendChild(emGroup);

    /* ---- error target ---- */
    const tGroup = group('误差目标', false);
    const tBody = tGroup.querySelector('.body');
    tBody.appendChild(field('要最小化/评估的误差', select([
      { value: 'alpha', label: '天然分馏因子 α' },
      { value: 'ratio', label: '指定同位素比值' },
    ], state.errorTarget, (v) => { state.errorTarget = v; buildControls(); render(); })));
    if (state.errorTarget === 'ratio') {
      const opts = sys().isonum.map((v) => ({ value: v, label: v + state.element }));
      tBody.appendChild(el('div', { class: 'iso-grid' }, [
        field('分子', select(opts, sys().isonum[state.erNum], (v) => { state.erNum = sys().isonum.indexOf(parseInt(v, 10)); render(); })),
        field('分母', select(opts, sys().isonum[state.erDen], (v) => { state.erDen = sys().isonum.indexOf(parseInt(v, 10)); render(); })),
      ]));
    }
    PANEL.appendChild(tGroup);

    /* ---- plot options ---- */
    const pGroup = group('图形设置', false);
    const pBody = pGroup.querySelector('.body');
    pBody.appendChild(field('二维图分辨率', numberInput(state.resolution, (v) => {
      state.resolution = Math.max(40, Math.min(400, Math.round(v)));
      render();
    }, '10'), '每边网格数。160 约 2.6 万个点，200 以上会明显变慢。'));
    pBody.appendChild(field('等高线阈值（相对最优误差）', numberInput(state.threshold, (v) => {
      state.threshold = Math.max(0.01, Math.min(5, v));
      render();
    }, '0.05')));
    PANEL.appendChild(pGroup);
  }

  function onElementChange() {
    resetStandard();
    const s = sys();
    state.isoinv = suggestedIsotopes();
    state.spikeType = s.rawspike.length >= 2 ? state.spikeType : 'pure';
    state.spikeA = 0;
    state.spikeB = 1;
    state.q = 0.5;
    state.erNum = 0;
    state.erDen = Math.min(1, s.isonum.length - 1);
    state.emod.radiogenic = isRadiogenic(state.element);
    buildSpike();
    buildControls();
    render();
  }

  /* ====================================================================== *
   * tab 1: precision
   * ====================================================================== */
  function renderPrecision() {
    const s = sys();
    const spike = state.spike;
    const alpha = 0, beta = 0;

    /* --- cards --- */
    const emod = buildEmod();
    const prep = prepFor(spike);
    let best = { p: 0.5, e: Infinity, ppm: Infinity };
    const NP = 41;
    const pp = [], ee = [], pm = [];
    for (let i = 0; i < NP; i++) {
      const p = 0.02 + (0.96 * i) / (NP - 1);
      const r = DS.errorEstimate(s, prep, emod, p, spike, alpha, beta);
      pp.push(p); ee.push(r.error); pm.push(r.ppmperamu);
      if (r.error < best.e) best = { p, e: r.error, ppm: r.ppmperamu };
    }
    const atHalf = DS.errorEstimate(s, prep, emod, 0.5, spike, alpha, beta);

    const cards = $('precCards');
    cards.innerHTML = '';
    const add = (k, v, u, hl) => cards.appendChild(el('div', { class: 'card' + (hl ? ' hl' : '') }, [
      el('div', { class: 'k' }, [k]),
      el('div', { class: 'v', html: v + (u ? '<span class="u">' + u + '</span>' : '') }),
    ]));
    add('最优混合比 p', pct(best.p), '', true);
    add('α 误差 (1SD)', fmt(best.e, 6), '');
    add('精度', fmt(best.ppm, 2), 'ppm/amu', true);
    add('p = 0.5 时的 α 误差', fmt(atHalf.error, 6), '');
    add('当前稀释剂', state.q === 0.5 ? '50 : 50' : pct(state.q, 1) + ' A', '');

    const yLab = state.errorTarget === 'alpha' ? 'α 误差 (1SD)' : s.isonum[state.erNum] + '/' + s.isonum[state.erDen] + ' 误差 (1SD)';

    /* --- 1D: error vs p --- */
    drawLineChart($('plotP'), [{ x: pp, y: ee, color: css('--accent') }], {
      height: 240,
      xLabel: '双稀释剂在混合样中的比例 p',
      yLabel: yLab,
      markers: [{ x: best.p, y: best.e, label: 'p* = ' + best.p.toFixed(3) }],
    });

    /* --- 1D: error vs spike composition q --- */
    const qs = [], es = [];
    let bestQ = { q: 0.5, e: Infinity };
    const s1 = new Float64Array(s.isonum.length), s2 = new Float64Array(s.isonum.length);
    if (state.spikeType === 'pure') {
      s1[state.spikeA] = 1; s2[state.spikeB] = 1;
    } else {
      for (let i = 0; i < s.isonum.length; i++) {
        s1[i] = s.rawspike[state.spikeA][i];
        s2[i] = s.rawspike[state.spikeB][i];
      }
    }
    for (let i = 0; i < 61; i++) {
      const q = 0.01 + (0.98 * i) / 60;
      const sp = DS.mix(s1, s2, q);
      const r = DS.errorEstimate(s, prep, emod, best.p, sp, alpha, beta);
      qs.push(q); es.push(r.error);
      if (r.error < bestQ.e) bestQ = { q, e: r.error };
    }
    const nameA = state.spikeType === 'pure'
      ? s.isonum[state.spikeA] + state.element
      : '单稀释剂 ' + (state.spikeA + 1);
    const nameB = state.spikeType === 'pure'
      ? s.isonum[state.spikeB] + state.element
      : '单稀释剂 ' + (state.spikeB + 1);
    drawLineChart($('plotQ'), [{ x: qs, y: es, color: css('--good') }], {
      height: 240,
      xLabel: 'A（' + nameA + '）在双稀释剂中的比例 q',
      yLabel: yLab,
      markers: [{ x: bestQ.q, y: bestQ.e, color: css('--warn'), label: 'q* = ' + bestQ.q.toFixed(3) }],
    });

    /* --- 2D map --- */
    const N = state.resolution;
    const emod2 = buildEmod();
    const meanSpike = DS.mix(s1, s2, 0.5);
    const prep2 = DS.prepareSystem(s, meanSpike, isoinvIndices(), errorRatioIndices(), emod2);
    const grid = new Float64Array(N * N);
    let vmin = Infinity;
    const xsArr = new Float64Array(N), ysArr = new Float64Array(N);
    for (let i = 0; i < N; i++) { xsArr[i] = 0.005 + (0.99 * i) / (N - 1); ysArr[i] = 0.005 + (0.99 * i) / (N - 1); }
    for (let j = 0; j < N; j++) {
      const q = ysArr[j];
      const sp = DS.mix(s1, s2, q);
      for (let i = 0; i < N; i++) {
        const r = DS.errorEstimate(s, prep2, emod2, xsArr[i], sp, alpha, beta);
        const v = state.errorTarget === 'alpha' ? r.error : r.ppmperamu;
        grid[j * N + i] = isFinite(v) ? v : Infinity;
        if (isFinite(v) && v < vmin) vmin = v;
      }
    }
    const vmax = vmin * (1 + state.threshold);
    drawHeatMap($('plotMap'), grid, N, N, xsArr[0], xsArr[N - 1], ysArr[0], ysArr[N - 1], vmin, vmax, {
      height: 380,
      xLabel: '双稀释剂在混合样中的比例 p',
      yLabel: 'A（' + nameA + '）在双稀释剂中的比例 q',
      marker: { x: best.p, y: bestQ.q },
      cbarLabel: yLab,
      vFormat: (v) => fmt(v, 3),
    });

    const note = $('mapNote');
    note.innerHTML = '颜色范围为 <b>' + fmt(vmin, 4) + '</b> 到 <b>' + fmt(vmax, 4) + '</b>（最优值的 ' +
      (1 + state.threshold).toFixed(2) + ' 倍）。超出范围的点留白。红点为最优组合：p = ' + best.p.toFixed(4) +
      '，q = ' + bestQ.q.toFixed(4) + '，误差 ' + fmt(vmin, 5) + '。';
  }

  /* ====================================================================== *
   * tab 2: optimal spikes
   * ====================================================================== */
  function renderOptimal(force) {
    const s = sys();
    const emod = buildEmod();
    const invIdx = isoinvIndices();
    const er = errorRatioIndices();
    const rows = [];

    if (state.spikeType === 'pure') {
      for (let a = 0; a < invIdx.length; a++) {
        for (let b = a + 1; b < invIdx.length; b++) {
          const s1 = new Float64Array(s.isonum.length), s2 = new Float64Array(s.isonum.length);
          s1[invIdx[a]] = 1; s2[invIdx[b]] = 1;
          const mean = DS.mix(s1, s2, 0.5);
          const prep = DS.prepareSystem(s, mean, invIdx, er, emod);
          const best = DS.optimalSpike(s, prep, emod, s1, s2, 0, 0, 11);
          const sp = DS.mix(s1, s2, best.q);
          const r = DS.errorEstimate(s, prep, emod, best.p, sp, 0, 0);
          rows.push({
            label: s.isonum[invIdx[a]] + '-' + s.isonum[invIdx[b]],
            spike: sp, prop: best.p, q: best.q, err: r.error, ppm: r.ppmperamu,
          });
        }
      }
    } else {
      const R = s.rawspike;
      for (let a = 0; a < R.length; a++) {
        for (let b = a + 1; b < R.length; b++) {
          let ok = true;
          for (const ix of invIdx) if (R[a][ix] + R[b][ix] <= 0) ok = false;
          if (!ok) continue;
          const mean = DS.mix(R[a], R[b], 0.5);
          const prep = DS.prepareSystem(s, mean, invIdx, er, emod);
          const best = DS.optimalSpike(s, prep, emod, R[a], R[b], 0, 0, 11);
          const sp = DS.mix(R[a], R[b], best.q);
          const r = DS.errorEstimate(s, prep, emod, best.p, sp, 0, 0);
          rows.push({
            label: '单稀释剂 ' + (a + 1) + ' + ' + (b + 1), spike: sp,
            prop: best.p, q: best.q, err: r.error, ppm: r.ppmperamu,
          });
        }
      }
    }

    rows.sort((x, y) => x.err - y.err);
    state.optimalRows = rows;

    const tbody = $('optBody');
    tbody.innerHTML = '';
    const head = $('optHead');
    head.innerHTML = '';
    ['同位素对', '最佳 p', '最佳 q', 'α 误差 (1SD)', 'ppm/amu', '稀释剂组成 (%)']
      .forEach((label) => head.appendChild(el('th', null, [label])));
    rows.forEach((row, i) => {
      const tr = el('tr', { class: 'clickable', onclick: () => applyOptimal(i) }, [
        el('td', { class: i === 0 ? 'best' : '' }, [row.label]),
        el('td', { html: fmt(row.prop, 5) }),
        el('td', { html: fmt(row.q, 5) }),
        el('td', { html: fmt(row.err, 6) }),
        el('td', { html: fmt(row.ppm, 3) }),
        el('td', { style: 'text-align:left' },
          [Array.from(row.spike).map((v) => (v > 1e-4 ? (v * 100).toFixed(2) : '·')).join('  ')]),
      ]);
      tbody.appendChild(tr);
    });
    $('optCount').textContent = rows.length + ' 种组合';
  }

  function applyOptimal(i) {
    const row = state.optimalRows[i];
    if (!row) return;
    state.q = row.q;
    if (state.spikeType === 'pure') {
      // recover the pair from the label
      const parts = row.label.split('-');
      state.spikeA = sys().isonum.indexOf(parseInt(parts[0], 10));
      state.spikeB = sys().isonum.indexOf(parseInt(parts[1], 10));
    } else {
      const nums = row.label.match(/\d+/g);
      state.spikeA = parseInt(nums[0], 10) - 1;
      state.spikeB = parseInt(nums[1], 10) - 1;
    }
    buildSpike();
    buildControls();
    render();
    showTab('precision');
  }

  /* ====================================================================== *
   * tab 3: inversion of measured data
   * ====================================================================== */
  function renderInvert() {
    // nothing to do until the user presses the button
  }

  function parseMeasurements(text) {
    const rows = [];
    text.split(/\r?\n/).forEach((line) => {
      const t = line.trim();
      if (!t || t.startsWith('#')) return;
      const parts = t.split(/[\s,;\t]+/).filter((v) => v.length);
      const nums = parts.map(Number);
      if (nums.length === sys().isonum.length && nums.every((v) => isFinite(v))) rows.push(Float64Array.from(nums));
    });
    return rows;
  }

  function doInversion() {
    const rows = parseMeasurements($('measInput').value);
    const s = sys();
    if (!rows.length) {
      $('invNote').innerHTML = '<div class="banner bad">没有解析到有效数据。每行需要 ' + s.isonum.length + ' 个数值，顺序为 ' + s.isonum.join(', ') + '。</div>';
      return;
    }
    const spike = state.spike;
    const inv = isoinvIndices();
    const tbody = $('invBody');
    tbody.innerHTML = '';
    const results = [];
    let worst = 0, nBad = 0;
    rows.forEach((row, i) => {
      const out = DS.reduce(s, row, spike, inv.slice(), state.standard);
      results.push(out);
      if (!out || !out.ok) nBad++;
      if (out) worst = Math.max(worst, out.residual);
      const cells = out ? [
        el('td', null, [String(i + 1)]),
        el('td', { html: fmt(out.alpha, 7) }),
        el('td', { html: fmt(out.beta, 5) }),
        el('td', { html: fmt(out.prop * 100, 4) }),
        el('td', { html: fmt(out.lambda, 5) }),
        el('td', { html: fmt(out.residual, 2), style: out.residual < 1e-9 ? '' : 'color:var(--warn)' }),
        el('td', { html: out.ok ? '<span style="color:var(--good)">收敛</span>' : '<span style="color:var(--bad)">未收敛</span>' }),
      ] : [
        el('td', null, [String(i + 1)]), el('td', null, ['—']), el('td', null, ['—']),
        el('td', null, ['—']), el('td', null, ['—']), el('td', null, ['—']),
        el('td', { html: '<span style="color:var(--bad)">失败</span>' }),
      ];
      tbody.appendChild(el('tr', null, cells));
    });
    state.results = { rows, results };

    const good = results.filter((r) => r && r.ok).map((r) => r.alpha);
    let summary = '';
    if (good.length > 1) {
      const mean = good.reduce((a, b) => a + b, 0) / good.length;
      const sd = Math.sqrt(good.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (good.length - 1));
      summary = `共 ${rows.length} 个测点，成功 ${good.length} 个。<b>α 均值 ${mean.toExponential(5)}</b>，` +
        `<b>1SD = ${sd.toExponential(4)}</b>。`;
    } else {
      summary = `共 ${rows.length} 个测点，成功 ${good.length} 个。`;
    }
    if (nBad) summary += ` <span style="color:var(--bad)">${nBad} 个测点未收敛</span>（方程残差 ${worst.toExponential(2)}），请检查稀释剂比例或反演同位素选择。`;
    $('invNote').innerHTML = '<div class="banner' + (nBad ? ' warn' : '') + '">' + summary + '</div>';

    $('invExport').disabled = false;
  }

  function exportInversion() {
    if (!state.results) return;
    const s = sys();
    const lines = ['#' + ['index', 'alpha', 'beta', 'prop', 'lambda'].concat(s.isonum.map((v) => 'sample_' + v + s.element)).join(',')];
    state.results.results.forEach((r, i) => {
      if (!r) { lines.push((i + 1) + ',' + new Array(9).fill('').join(',')); return; }
      lines.push([i + 1, r.alpha, r.beta, r.prop, r.lambda].concat(Array.from(r.sample)).join(','));
    });
    download(lines.join('\n'), 'doublespike-inversion.csv');
  }

  function download(text, name) {
    const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  function runMonteCarlo() {
    const s = sys();
    const emod = buildEmod();
    const n = parseInt($('mcN').value, 10) || 300;
    const prop = parseFloat($('mcProp').value);
    const alpha = parseFloat($('mcAlpha').value) || 0;
    const beta = parseFloat($('mcBeta').value) || 0;
    const seed = parseInt($('mcSeed').value, 10) || 1;

    const rows = DS.monteRun(s, emod, prop, state.spike, alpha, beta, n, seed);
    const inv = isoinvIndices();
    const alphas = [], props = [];
    let nBad = 0;
    rows.forEach((row) => {
      const out = DS.reduce(s, row, state.spike, inv.slice(), state.standard);
      if (out && out.ok) { alphas.push(out.alpha); props.push(out.prop); }
      else nBad++;
    });
    const mean = alphas.reduce((a, b) => a + b, 0) / alphas.length;
    const sd = Math.sqrt(alphas.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (alphas.length - 1));
    const prep = prepFor(state.spike);
    const pred = DS.errorEstimate(s, prep, emod, prop, state.spike, alpha, beta);

    $('mcOut').innerHTML =
      `<div class="cards" style="margin:0">
        <div class="card"><div class="k">蒙特卡洛 α 均值</div><div class="v">${fmt(mean, 6)}</div></div>
        <div class="card hl"><div class="k">蒙特卡洛 1SD</div><div class="v">${fmt(sd, 6)}</div></div>
        <div class="card"><div class="k">线性误差传播预测</div><div class="v">${fmt(pred.error, 6)}</div></div>
        <div class="card"><div class="k">比值 SD / 预测</div><div class="v">${(sd / pred.error).toFixed(3)}</div></div>
       </div>
       <div class="hint" style="margin-top:8px">${alphas.length} 次成功反演${nBad ? '，' + nBad + ' 次失败' : ''}。
       比值接近 1 说明线性误差传播是可靠的；明显偏离说明该稀释剂在该比例下线性近似失效。</div>`;
  }

  /* ====================================================================== *
   * tabs
   * ====================================================================== */
  function showTab(name) {
    state.tab = name;
    document.querySelectorAll('nav.tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
    document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('on', p.id === 'panel-' + name));
    if (name === 'optimal') renderOptimal();
    if (name === 'precision') render();
    if (name === 'invert') ensureSampleData();
  }

  /* ====================================================================== *
   * master render
   * ====================================================================== */
  let renderToken = 0;

  /**
   * Defer work just long enough for the browser to paint the "计算中" badge.
   *
   * NB: requestAnimationFrame is *not* used here on purpose.  It never fires in
   * a headless browser and is throttled to nothing in a background tab, which
   * would leave the page permanently unrendered in those cases.
   */
  function defer(work, delay) {
    setTimeout(work, delay === undefined ? 16 : delay);
  }

  function render() {
    const token = ++renderToken;
    const busy = $('busy');
    if (busy) busy.style.display = '';
    defer(() => {
      try {
        buildSpike();
        if (state.tab === 'precision') renderPrecision();
        if (state.tab === 'optimal') renderOptimal();
      } catch (err) {
        console.error(err);
        window.__lastRenderError = String(err && err.message ? err.message : err);
        const n = $('mapNote');
        if (n) n.innerHTML = '<span style="color:var(--bad)">计算出错：' + err.message + '</span>';
      } finally {
        if (token === renderToken && busy) busy.style.display = 'none';
      }
    });
  }

  /* ====================================================================== *
   * self test
   * ====================================================================== */
  const SELF_TEST_CASES = [
    // values computed with the Python package, see webgui/src/reference.json
    { name: 'Fe α 误差', want: 0.0036364520647345615, tol: 1e-12,
      got: () => {
        const s = ELEMENTS.Fe;
        const emod = DS.buildErrorModel(4, {});
        const inv = [0, 1, 2, 3];
        const spike = new Float64Array([0, 0, 0.5, 0.5]);
        const prep = DS.prepareSystem(s, spike, inv, null, emod);
        return DS.errorEstimate(s, prep, emod, 0.5, spike, -0.2, 1.8).error;
      } },
    { name: 'Fe ppm/amu', want: 64.72189557894438, tol: 1e-10,
      got: () => {
        const s = ELEMENTS.Fe;
        const emod = DS.buildErrorModel(4, {});
        const spike = new Float64Array([0, 0, 0.5, 0.5]);
        const prep = DS.prepareSystem(s, spike, [0, 1, 2, 3], null, emod);
        return DS.errorEstimate(s, prep, emod, 0.5, spike, -0.2, 1.8).ppmperamu;
      } },
    { name: 'Fe 正演–反演闭合 (α)', want: -0.2, tol: 1e-9,
      got: () => {
        const s = ELEMENTS.Fe;
        const sp = new Float64Array([0, 0, 0.5, 0.5]);
        const sample = DS.normalise(Array.from(s.standard, (v, i) => v * Math.pow(s.mass[i], 0.2)));
        const mixc = Array.from(s.standard, (v, i) => 0.5 * sp[i] + 0.5 * sample[i]);
        const meas = DS.normalise(Array.from(mixc, (v, i) => v * Math.pow(s.mass[i], 1.7)));
        return DS.reduce(s, Float64Array.from(meas), sp, [0, 1, 2, 3], s.standard).alpha;
      } },
    { name: 'Fe 反演回收 β', want: 1.7, tol: 1e-9,
      got: () => {
        const s = ELEMENTS.Fe;
        const sp = new Float64Array([0, 0, 0.5, 0.5]);
        const sample = DS.normalise(Array.from(s.standard, (v, i) => v * Math.pow(s.mass[i], 0.2)));
        const mixc = Array.from(s.standard, (v, i) => 0.5 * sp[i] + 0.5 * sample[i]);
        const meas = DS.normalise(Array.from(mixc, (v, i) => v * Math.pow(s.mass[i], 1.7)));
        return DS.reduce(s, Float64Array.from(meas), sp, [0, 1, 2, 3], s.standard).beta;
      } },
    { name: '五个同位素体系也能反演 (Ge)', want: 0.15, tol: 1e-8,
      got: () => {
        const s = ELEMENTS.Ge;
        const sp = new Float64Array(5); sp[0] = 0.72; sp[2] = 0.28;
        const inv = [0, 1, 2, 3];
        const sample = DS.normalise(Array.from(s.standard, (v, i) => v * Math.pow(s.mass[i], -0.15)));
        const mixc = Array.from(s.standard, (v, i) => 0.4 * sp[i] + 0.6 * sample[i]);
        const meas = DS.normalise(Array.from(mixc, (v, i) => v * Math.pow(s.mass[i], 0.9)));
        const out = DS.reduce(s, Float64Array.from(meas), sp, inv, s.standard);
        return out.ok ? out.alpha : NaN;
      } },
  ];

  function canvasIsBlank(id) {
    const c = $(id);
    if (!c) return 'no element';
    if (!c.width || !c.height) return `size 0 (clientWidth=${c.clientWidth})`;
    const ctx = c.getContext('2d');
    const w = Math.min(c.width, 240), h = Math.min(c.height, 160);
    const d = ctx.getImageData(0, 0, w, h).data;
    let ink = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] !== 0) ink++;
    return ink === 0 ? `${c.width}x${c.height} blank, clientWidth=${c.clientWidth}` : null;
  }

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  async function runSelfTest() {
    const out = [];
    SELF_TEST_CASES.forEach((c) => {
      let got, ok = false, err = '';
      try {
        got = c.got();
        ok = isFinite(got) && Math.abs(got - c.want) <= c.tol;
      } catch (e) { err = e.message; }
      out.push({ name: c.name, ok, detail: ok ? '' : `got ${got} want ${c.want}${err ? ' (' + err + ')' : ''}` });
    });

    // the three tabs must render without throwing and must produce pixels.
    // Rendering is asynchronous (it defers to the next animation frame so the
    // "计算中" badge can appear), hence the awaits.
    for (const tab of ['precision', 'optimal', 'invert']) {
      try {
        showTab(tab);
        await wait(120);
        if (tab === 'precision') {
          await wait(600);
          const b1 = canvasIsBlank('plotP'), b2 = canvasIsBlank('plotQ'), b3 = canvasIsBlank('plotMap');
          out.push({ name: '精度图：一维曲线已绘制', ok: !b1 && !b2,
            detail: [b1, b2].filter(Boolean).join('; ') });
          out.push({ name: '精度图：二维误差图已绘制', ok: !b3, detail: b3 || '' });
        }
        if (tab === 'optimal') {
          out.push({ name: '最优稀释剂：表格有内容', ok: $('optBody').children.length > 0,
            detail: $('optBody').children.length + ' 行' });
        }
        if (tab === 'invert') {
          buildSampleData();
          doInversion();
          const rows = $('invBody').children.length;
          const ok = Array.from($('invBody').querySelectorAll('td')).some((td) => td.textContent === '收敛');
          out.push({ name: '数据反演：示例数据全部收敛', ok: ok && rows === 3, detail: 'rows=' + rows });
          // the demo analyses must come back with their own generating parameters
          const rs = state.results ? state.results.results : [];
          const diffs = rs.map((r, i) => r ? Math.max(
            Math.abs(r.alpha - SAMPLE_TRUTH[i].a),
            Math.abs(r.beta - SAMPLE_TRUTH[i].b),
            Math.abs(r.prop - SAMPLE_TRUTH[i].p)) : Infinity);
          const recovered = diffs.length === SAMPLE_TRUTH.length && diffs.every((d) => d < 1e-7);
          out.push({ name: '数据反演：找回生成参数 (p, α, β)', ok: recovered,
            detail: recovered ? '' : 'max diff ' + Math.max(...diffs).toExponential(2) + ' [' +
              rs.map((r, i) => r ? `${r.prop.toFixed(4)}/${r.alpha.toFixed(4)}/${r.beta.toFixed(3)} vs ${SAMPLE_TRUTH[i].p}/${SAMPLE_TRUTH[i].a}/${SAMPLE_TRUTH[i].b}` : 'null').join('  ') + ']' });
        }
        out.push({ name: '切换到「' + tab + '」无异常', ok: true, detail: '' });
      } catch (e) {
        out.push({ name: '切换到「' + tab + '」', ok: false, detail: e.message });
      }
    }
    showTab('precision');
    await wait(200);

    const nPass = out.filter((r) => r.ok).length;
    const html = '<div class="banner ' + (nPass === out.length ? '' : 'bad') + '">' +
      '<b>自检 ' + nPass + '/' + out.length + ' 通过</b>' +
      '<ul style="margin:6px 0 0;padding-left:18px">' +
      out.map((r) => '<li>' + (r.ok ? '✓' : '✗') + ' ' + r.name +
        (r.detail ? ' <span style="color:var(--text-faint)">' + r.detail + '</span>' : '') + '</li>').join('') +
      '</ul></div>';
    $('selftestOut').innerHTML = html;
    document.title = 'selftest ' + nPass + '/' + out.length;
    return { passed: nPass, total: out.length, results: out };
  }

  /* ====================================================================== *
   * boot
   * ====================================================================== */
  function init() {
    // theme
    const saved = localStorage.getItem('ds-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    $('theme').addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('ds-theme', next);
      render();
      if (state.tab === 'optimal') renderOptimal();
    });

    // tabs
    document.querySelectorAll('nav.tabs button').forEach((b) => {
      b.addEventListener('click', () => showTab(b.dataset.tab));
    });

    // inversion buttons
    $('invRun').addEventListener('click', doInversion);
    $('invExport').addEventListener('click', exportInversion);
    $('invFile').addEventListener('change', (ev) => {
      const f = ev.target.files[0];
      if (!f) return;
      const rd = new FileReader();
      rd.onload = () => { $('measInput').value = rd.result; };
      rd.readAsText(f);
    });
    $('mcRun').addEventListener('click', runMonteCarlo);
    $('selftest').addEventListener('click', runSelfTest);

    resetStandard();
    state.isoinv = suggestedIsotopes();
    state.spikeA = 0; state.spikeB = 1;
    state.emod.radiogenic = isRadiogenic(state.element);
    buildSpike();
    buildControls();
    buildSampleData();
    render();

    if (location.hash.indexOf('selftest') >= 0) {
      setTimeout(runSelfTest, 30);
    } else {
      // allow deep links such as #tab=optimal
      const m = /(?:^|[#&])tab=([a-z]+)/i.exec(location.hash);
      if (m && ['precision', 'optimal', 'invert', 'theory'].includes(m[1].toLowerCase())) {
        showTab(m[1].toLowerCase());
      }
    }

    // redraw on resize
    let rt;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => { render(); if (state.tab === 'optimal') { /* table only */ } }, 200);
    });
  }

  /** A few synthetic analyses so that the inversion tab is never empty. */
  const SAMPLE_TRUTH = [
    { p: 0.42, a: -0.12, b: 1.6 },
    { p: 0.55, a: 0.08, b: -0.9 },
    { p: 0.35, a: 0.21, b: 2.4 },
  ];

  function syntheticBeams(t) {
    const s = sys();
    // the sample composition must be normalised before it is mixed, otherwise
    // the mole fraction of spike in the mixture is not (1 - p) any more
    const sample = DS.normalise(
      Array.from(s.isonum, (_, i) => s.standard[i] * Math.pow(s.mass[i], -t.a))
    );
    const mixc = Array.from(s.isonum, (_, i) => t.p * state.spike[i] + (1 - t.p) * sample[i]);
    const norm = mixc.reduce((x, y) => x + y, 0);
    return mixc.map((v, i) => (v / norm) * Math.pow(s.mass[i], t.b) * 10);
  }

  function buildSampleData() {
    const s = sys();
    // 10 significant digits, not a fixed number of decimals: for a strongly
    // fractionated run some beams are small, and rounding them to 6 decimals
    // perturbs the inversion by far more than the solver's own tolerance.
    $('measInput').value = SAMPLE_TRUTH
      .map((t) => syntheticBeams(t).map((v) => v.toPrecision(10)).join(', '))
      .join('\n');
    const cols = s.isonum.map((v) => v + state.element).join(', ');
    $('measHint').innerHTML = '每行一个测点，' + s.isonum.length + ' 列，依次为 <b>' + cols +
      '</b>（束流强度或任意正比量，程序内部会归一化）。可以逗号、空格或制表符分隔；以 # 开头的行会被忽略。';
  }

  /** Load the demo analyses and run them the first time the tab is opened. */
  function ensureSampleData() {
    if (!$('measInput').value.trim()) buildSampleData();
    if (!state.results) doInversion();
  }

  window.addEventListener('DOMContentLoaded', init);
})();
