/*
 * Verify that the JavaScript port in webgui/src/math.js reproduces the Python
 * library to the last few digits.  Reference values are produced by
 * tools/make_webgui_data.py (webgui/src/reference.json).
 *
 *   node tools/test_webgui_math.js
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = process.argv[2] || path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'webgui', 'src');

// load data.js (plain script defining const ELEMENTS / ELEMENT_ORDER) and math.js
const sandbox = { window: {}, module: undefined, console };
vm.createContext(sandbox);
for (const f of ['data.js', 'math.js']) {
  const code = fs.readFileSync(path.join(SRC, f), 'utf8');
  vm.runInContext(code, sandbox, { filename: f });
}
const ELEMENTS = vm.runInContext('ELEMENTS', sandbox);
const ELEMENT_ORDER = vm.runInContext('ELEMENT_ORDER', sandbox);
const DS = sandbox.window.DS;
if (!DS) throw new Error('math.js did not define window.DS');

const ref = JSON.parse(fs.readFileSync(path.join(SRC, 'reference.json'), 'utf8'));

let pass = 0, fail = 0;
const worst = {};

function check(label, got, want, tol) {
  const d = Math.abs(got - want) / Math.max(Math.abs(want), 1e-300);
  worst[label] = Math.max(worst[label] || 0, d);
  if (!(d <= tol)) {
    fail++;
    console.log(`  FAIL ${label}: js=${got} py=${want} rel=${d.toExponential(2)}`);
  } else {
    pass++;
  }
}

/* ------------------------------------------------------------------ */
console.log('1. error estimates');
for (const c of ref.error_estimates) {
  const sys = ELEMENTS[c.element];
  const isoinv = DS.isotopeIndices(sys, c.isoinv);
  const emod = DS.buildErrorModel(sys.isonum.length, {
    intensity: 10, deltat: 8, R: 1e11, T: 300,
    measuredType: 'fixed-total',
    radiogenic: ['Pb', 'Sr', 'Hf', 'Os', 'Nd'].includes(c.element),
  });
  // the error on alpha
  const prep = DS.prepareSystem(sys, c.spike, isoinv, null, emod);
  const out = DS.errorEstimate(sys, prep, emod, c.prop, c.spike, c.alpha, c.beta);
  check(`${c.element} error`, out.error, c.error, 1e-11);
  check(`${c.element} ppm/amu`, out.ppmperamu, c.ppmperamu, 1e-11);
  // the error on a chosen ratio (a separate preparation, different denominator)
  if (c.error_ratio !== null) {
    const er = DS.isotopeIndices(sys, c.errorratio);
    const prep2 = DS.prepareSystem(sys, c.spike, isoinv, er, emod);
    const out2 = DS.errorEstimate(sys, prep2, emod, c.prop, c.spike, c.alpha, c.beta);
    check(`${c.element} ratio error`, out2.error, c.error_ratio, 1e-10);
    check(`${c.element} ratio ppm/amu`, out2.ppmperamu, c.ppmperamu_ratio, 1e-10);
  }
}

/* ------------------------------------------------------------------ */
console.log('2. inversion');
for (const c of ref.inversions) {
  const sys = ELEMENTS[c.element];
  const isoinv = DS.isotopeIndices(sys, c.isoinv);
  const out = DS.reduce(sys, c.measured, c.spike, isoinv.slice(), sys.standard);
  if (!out) { fail++; console.log(`  FAIL ${c.element}: no solution`); continue; }
  check(`${c.element} alpha`, out.alpha, c.alpha, 1e-8);
  check(`${c.element} beta`, out.beta, c.beta, 1e-8);
  check(`${c.element} prop`, out.prop, c.prop, 1e-7);
  if (!out.ok) { fail++; console.log(`  FAIL ${c.element}: not converged, residual ${out.residual}`); }
  else pass++;
}

/* ------------------------------------------------------------------ */
console.log('3. Monte Carlo scatter vs the linear prediction');
{
  // The two implementations use different RNGs, so this is a statistical check
  // rather than a digit for digit one: the scatter of the inverted alpha must
  // match the linear error propagation to within a few percent.
  const sys = ELEMENTS['Fe'];
  const emod = DS.buildErrorModel(sys.isonum.length, {});
  const spike = new Float64Array([0, 0, 0.5, 0.5]);
  const inv = [0, 1, 2, 3];
  const prop = 0.5, alpha = -0.2, beta = 1.8;
  const rows = DS.monteRun(sys, emod, prop, spike, alpha, beta, 4000, 1234);
  const got = [];
  let nBad = 0;
  for (const r of rows) {
    const out = DS.reduce(sys, r, spike, inv.slice(), sys.standard);
    if (out && out.ok) got.push(out.alpha); else nBad++;
  }
  const mean = got.reduce((a, b) => a + b, 0) / got.length;
  const sd = Math.sqrt(got.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (got.length - 1));
  const prep = DS.prepareSystem(sys, spike, inv, null, emod);
  const want = DS.errorEstimate(sys, prep, emod, prop, spike, alpha, beta).error;
  console.log(`   Monte Carlo 1SD ${sd.toExponential(4)}, linear prediction ${want.toExponential(4)}, ` +
              `ratio ${(sd / want).toFixed(3)} (${nBad} failed inversions)`);
  check('monte carlo vs linear propagation', sd / want, 1.0, 0.08);
  check('monte carlo inversion failures', nBad === 0 ? 0 : 1, 0, 0);
}

/* ------------------------------------------------------------------ */
console.log('4. optimal spike search');
{
  // reference values computed with doublespike.optimalspike for Fe 57-58
  const sys = ELEMENTS['Fe'];
  const isoinv = [0, 1, 2, 3];
  const emod = DS.buildErrorModel(4, {});
  const s1 = new Float64Array([0, 0, 1, 0]);
  const s2 = new Float64Array([0, 0, 0, 1]);
  const prep = DS.prepareSystem(sys, DS.mix(s1, s2, 0.5), isoinv, null, emod);
  const best = DS.optimalSpike(sys, prep, emod, s1, s2, 0, 0, 11);
  console.log(`   57Fe-58Fe: p=${best.p.toFixed(5)} q=${best.q.toFixed(5)} err=${best.e.toExponential(6)}`);
  check('optimal p (57-58)', best.p, 0.44861, 2e-5);
  check('optimal q (57-58)', best.q, 0.47648, 2e-5);
  check('optimal error (57-58)', best.e, 3.510080e-3, 1e-6);
}

console.log('');
const summary = Object.entries(worst).map(([k, v]) => `${k}=${v.toExponential(1)}`).join('  ');
console.log('worst relative deviations:', summary);
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
