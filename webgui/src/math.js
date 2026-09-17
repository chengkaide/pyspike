/*
 * Core numerics of the double spike toolbox, in plain JavaScript.
 *
 * This is a direct port of the Python package so that the browser GUI and the
 * command line library agree.  Every equation number refers to
 * Rudge, Reynolds & Bourdon (2009), Chem. Geol. 265:420-431, "The double spike
 * toolbox".
 *
 * Conventions
 * -----------
 *  - a "composition" is a normalised array of isotope abundances, one entry per
 *    isotope, in the order of ELEMENTS[el].isonum;
 *  - a "ratio" array always has the denominator isotope removed, so it has
 *    nisos - 1 entries;
 *  - `AP` is the array of log(mass_i / mass_denominator);
 *  - indices (0 based) are used internally, isotope *numbers* (56, 57, ...) are
 *    only used in the user interface.
 */

'use strict';

/* ========================================================================== *
 * 1. small dense linear algebra
 * ========================================================================== */

function zeros(n, m) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = new Float64Array(m === undefined ? n : m);
  return a;
}

function identity(n) {
  const a = zeros(n, n);
  for (let i = 0; i < n; i++) a[i][i] = 1;
  return a;
}

function matMul(A, B) {
  const n = A.length, k = B.length, m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let t = 0; t < k; t++) {
      const a = Ai[t];
      if (a === 0) continue;
      const Bt = B[t];
      for (let j = 0; j < m; j++) Ci[j] += a * Bt[j];
    }
  }
  return C;
}

function matVec(A, v) {
  const n = A.length, m = v.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += A[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

function transpose(A) {
  const n = A.length, m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

function sandwich(A, V) {
  // A V A^T
  return matMul(matMul(A, V), transpose(A));
}

function matAdd(A, B) {
  const n = A.length, m = A[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j];
  return C;
}

/** Solve A x = B for small dense systems (Gauss-Jordan with partial pivoting). */
function solve(Ain, Bin) {
  const n = Ain.length;
  const m = Bin[0].length;
  const A = Ain.map((row) => Float64Array.from(row));
  const B = Bin.map((row) => Float64Array.from(row));
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(A[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r][col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) return null; // singular
    if (piv !== col) {
      const t = A[piv]; A[piv] = A[col]; A[col] = t;
      const t2 = B[piv]; B[piv] = B[col]; B[col] = t2;
    }
    const d = A[col][col];
    for (let j = 0; j < n; j++) A[col][j] /= d;
    for (let j = 0; j < m; j++) B[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r][col];
      if (f === 0) continue;
      for (let j = 0; j < n; j++) A[r][j] -= f * A[col][j];
      for (let j = 0; j < m; j++) B[r][j] -= f * B[col][j];
    }
  }
  return B;
}

/* ========================================================================== *
 * 2. composition and ratio helpers
 * ========================================================================== */

function sum(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

function normalise(comp) {
  const s = sum(comp);
  const out = new Float64Array(comp.length);
  for (let i = 0; i < comp.length; i++) out[i] = comp[i] / s;
  return out;
}

function indexOfIsotope(sys, isonum) {
  const k = sys.isonum.indexOf(isonum);
  return k < 0 ? isonum : k;
}

function isotopeIndices(sys, isonums) {
  return isonums.map((v) => indexOfIsotope(sys, v));
}

/** Ratios of a composition with a given denominator index (denominator removed). */
function ratiosOf(comp, deno, nisos) {
  const out = new Float64Array(nisos - 1);
  let k = 0;
  for (let i = 0; i < nisos; i++) if (i !== deno) out[k++] = comp[i] / comp[deno];
  return out;
}

function ratioIndices(deno, nisos) {
  const out = [];
  for (let i = 0; i < nisos; i++) if (i !== deno) out.push(i);
  return out;
}

/** Index of the ratio num/deno inside the ratio array whose denominator is deno. */
function ratioIndex(num, deno) {
  return num < deno ? num : num - 1;
}

/** Build a composition from ratios with the given denominator set to 1. */
function compositionOf(ratios, deno, nisos) {
  const comp = new Float64Array(nisos).fill(1);
  let k = 0;
  for (let i = 0; i < nisos; i++) if (i !== deno) comp[i] = ratios[k++];
  return normalise(comp);
}

/* equation (9): proportion in ratio space <-> proportion per mole */
function ratioPropToRealProp(lambda, ratioA, ratioB) {
  const a = 1 + sum(ratioA);
  const b = 1 + sum(ratioB);
  return (lambda * a) / (lambda * a + (1 - lambda) * b);
}

function realPropToRatioProp(prop, ratioA, ratioB) {
  const a = 1 + sum(ratioA);
  const b = 1 + sum(ratioB);
  return (prop * b) / (prop * b + (1 - prop) * a);
}

/* ========================================================================== *
 * 3. the error model
 * ========================================================================== */

/* Fundamental constants, identical to the Python package. */
const ELEMENTARY_CHARGE = 1.60217646e-19;
const BOLTZMANN = 1.3806504e-23;

/**
 * Build the three sub-models (measured, spike, standard) from beam parameters.
 * This is equation (35) plus the counting statistics term.
 */
function buildErrorModel(nisos, opts) {
  const {
    intensity = 10,
    deltat = 8,
    R = 1e11,
    T = 300,
    measuredType = 'fixed-total',
    radiogenic = false,
    Rreference = 1e11,
  } = opts || {};

  const Rarr = Array.isArray(R) ? R : new Array(nisos).fill(R);
  const a = new Float64Array(nisos), b = new Float64Array(nisos);
  // equation (35), with the generalisation to different resistors per beam
  for (let i = 0; i < nisos; i++) a[i] = (4 * BOLTZMANN * T * Rreference * Rreference) / (deltat * Rarr[i]);
  const bv = (ELEMENTARY_CHARGE * Rreference) / deltat;
  for (let i = 0; i < nisos; i++) b[i] = bv;

  const zero = { a: new Float64Array(nisos), b: new Float64Array(nisos), c: new Float64Array(nisos) };
  const full = { a, b, c: new Float64Array(nisos) };
  return {
    measured: Object.assign({ type: measuredType, intensity }, full),
    spike: Object.assign({ type: 'fixed-total', intensity }, zero),
    standard: Object.assign(
      { type: 'fixed-total', intensity },
      radiogenic ? full : zero
    ),
  };
}

/** Covariance matrix of the ratios for a composition and an error model. */
function calcRatioCov(composition, emod, deno, prop) {
  const nisos = composition.length;
  const comp = normalise(composition);
  const total = sum(comp.filter((_, i) => true)); // sum over all beams
  const meanbeams = new Float64Array(nisos);
  for (let i = 0; i < nisos; i++) {
    meanbeams[i] = (emod.intensity * comp[i]) / total;
    if (emod.type === 'fixed-sample') meanbeams[i] /= 1 - prop;
  }
  const beamvar = new Float64Array(nisos);
  for (let i = 0; i < nisos; i++) {
    const mb = meanbeams[i];
    beamvar[i] = emod.a[i] + emod.b[i] * mb + emod.c[i] * mb * mb; // equation (34)
  }
  return covBeamToRatio(meanbeams, beamvar, deno);
}

function covBeamToRatio(meanbeams, beamvar, deno) {
  const nisos = meanbeams.length;
  const ni = ratioIndices(deno, nisos);
  const ii = ni.concat([deno]); // move the denominator to the end
  const k = ni.length;
  const M = zeros(k + 1, k + 1);
  for (let r = 0; r < ii.length; r++) M[r][r] = beamvar[ii[r]];
  const d = meanbeams[deno];
  const A = zeros(k, k + 1);
  for (let r = 0; r < k; r++) {
    A[r][r] = 1 / d;
    A[r][k] = -meanbeams[ni[r]] / (d * d); // equation (38)
  }
  return sandwich(A, M);
}

/* ========================================================================== *
 * 4. linear error propagation (appendix A)
 * ========================================================================== */

/**
 * Everything the error estimate needs that does not depend on the trial
 * (p, q) point: the denominator choice, the ratio bookkeeping and the
 * covariance matrices of the standard and of the spike.
 */
function prepareSystem(sys, spike, isoinvIndices, errorRatioIndices, emod) {
  const nisos = sys.isonum.length;
  const spikeN = normalise(spike);
  // denominator = the isotope the spike contains most of
  let deno = isoinvIndices[0], best = -1;
  for (const ix of isoinvIndices) {
    if (spikeN[ix] > best) { best = spikeN[ix]; deno = ix; }
  }
  const nume = isoinvIndices.filter((ix) => ix !== deno);
  const order = [deno].concat(nume); // inversion ratio order

  // invrat: positions of order[1..3] inside the full ratio array
  const invrat = order.slice(1).map((v) => ratioIndex(v, deno));

  const APfull = new Float64Array(nisos - 1);
  const An = new Float64Array(nisos - 1);
  const ni = ratioIndices(deno, nisos);
  for (let k = 0; k < ni.length; k++) {
    APfull[k] = Math.log(sys.mass[ni[k]] / sys.mass[deno]);
    An[k] = sys.standard[ni[k]] / sys.standard[deno];
  }

  const VAn = calcRatioCov(sys.standard, emod.standard, deno, 0);

  return {
    nisos, deno, invrat, order, APfull, An, VAn, spike: spikeN,
    errorRatioIndices,
  };
}

/**
 * Error on alpha (or on a chosen ratio) for one spike-sample proportion and one
 * spike composition.
 *
 * Returns { error, ppmperamu, lambda, prop }.
 */
function errorEstimate(sys, prep, emod, prop, spike, alpha, beta) {
  const { nisos, deno, invrat, APfull, An, VAn } = prep;
  const spikeN = normalise(spike);

  const AT = ratiosOf(spikeN, deno, nisos);
  const AN = new Float64Array(An.length);
  for (let i = 0; i < An.length; i++) AN[i] = An[i] * Math.exp(-alpha * APfull[i]);

  const lambda = realPropToRatioProp(prop, AT, AN);
  const AM = new Float64Array(AT.length);
  for (let i = 0; i < AT.length; i++) AM[i] = lambda * AT[i] + (1 - lambda) * AN[i];
  const Am = new Float64Array(AT.length);
  for (let i = 0; i < AT.length; i++) Am[i] = AM[i] * Math.exp(beta * APfull[i]);

  const measured = compositionOf(Am, deno, nisos);
  const VAT = calcRatioCov(spikeN, emod.spike, deno, prop);
  const VAm = calcRatioCov(measured, emod.measured, deno, prop);

  const z = [lambda, alpha, beta];
  const out = propagate(z, APfull, An, AT, Am, VAn, VAT, VAm, invrat);

  /* ---- error to report ---- */
  const nratios = An.length;
  if (!prep.errorRatioIndices) {
    const error = Math.sqrt(Math.max(out.Vz[1][1], 0));
    const meanMass = sum(sys.mass) / nisos;
    return { error, ppmperamu: (1e6 * error) / meanMass, lambda, prop }; // equation (51)
  }
  const [erNum, erDen] = prep.errorRatioIndices;
  const newVAN = changeDenomCov(AN, out.VAN, deno, erDen);
  const newAni = [];
  for (let i = 0; i < nisos; i++) if (i !== erDen) newAni.push(i);
  const erat = newAni.indexOf(erNum);
  const error = Math.sqrt(Math.max(newVAN[erat][erat], 0));
  const stdRatio = sys.standard[erNum] / sys.standard[erDen];
  const massDiff = Math.abs(sys.mass[erNum] - sys.mass[erDen]);
  return {
    error, ppmperamu: (1e6 * error) / (stdRatio * massDiff), lambda, prop,
  }; // equation (46)
}

/** Change the denominator of a covariance matrix of ratios. */
function changeDenomCov(data, datacov, olddi, newdi) {
  const nisos = data.length + 1;
  const oldni = [];
  for (let i = 0; i < nisos; i++) if (i !== olddi) oldni.push(i);
  const dataplus = new Float64Array(nisos);
  let k = 0;
  for (let i = 0; i < nisos; i++) dataplus[i] = i === olddi ? 1 : data[k++];
  const newni = [];
  for (let i = 0; i < nisos; i++) if (i !== newdi) newni.push(i);

  const datacovplus = zeros(nisos, nisos);
  for (let i = 0; i < oldni.length; i++)
    for (let j = 0; j < oldni.length; j++)
      datacovplus[oldni[i]][oldni[j]] = datacov[i][j];

  const den = dataplus[newdi];
  const A = identity(nisos);
  for (let i = 0; i < nisos; i++) A[i][i] = 1 / den;
  for (let i = 0; i < nisos; i++) A[i][newdi] -= dataplus[i] / (den * den);

  const M = sandwich(A, datacovplus);
  const out = zeros(newni.length, newni.length);
  for (let i = 0; i < newni.length; i++)
    for (let j = 0; j < newni.length; j++) out[i][j] = M[newni[i]][newni[j]];
  return out;
}

/** Derivatives of z = (lambda, alpha, beta) with respect to n, T, m. */
function zSensitivity(z, P, n, T, m) {
  const lambda = z[0], alpha = z[1], beta = z[2];
  const k = P.length;
  const N = new Float64Array(k), M = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    N[i] = n[i] * Math.exp(-P[i] * alpha);
    M[i] = m[i] * Math.exp(-P[i] * beta);
  }

  const dfdy = zeros(3, 3);
  for (let i = 0; i < k; i++) {
    dfdy[i][0] = T[i] - N[i] * (1 + alpha * P[i]); // equation (15)
    dfdy[i][1] = -N[i] * P[i];
    dfdy[i][2] = M[i] * P[i];
  }
  const dfdT = zeros(k, k), dfdm = zeros(k, k), dfdn = zeros(k, k);
  for (let i = 0; i < k; i++) {
    dfdT[i][i] = lambda; // equation (20)
    dfdm[i][i] = -Math.exp(-beta * P[i]);
    dfdn[i][i] = (1 - lambda) * Math.exp(-alpha * P[i]);
  }
  // equation (22): (lambda, (1-lambda)alpha, beta) -> (lambda, alpha, beta)
  const K = [[1, 0, 0], [alpha / (1 - lambda), 1 / (1 - lambda), 0], [0, 0, 1]];

  const rhs = solve(dfdy, concatCols([dfdT, dfdm, dfdn]));
  if (!rhs) return null;
  const split = splitCols(rhs, k);
  return {
    dzdn: negMatMul(K, split[2]), // equation (17)
    dzdm: negMatMul(K, split[1]), // equation (18)
    dzdT: negMatMul(K, split[0]), // equation (19)
  };
}

function concatCols(mats) {
  const n = mats[0].length;
  const out = zeros(n, sum(mats.map((m) => m[0].length)));
  let off = 0;
  for (const m of mats) {
    for (let i = 0; i < n; i++) for (let j = 0; j < m[0].length; j++) out[i][off + j] = m[i][j];
    off += m[0].length;
  }
  return out;
}

function splitCols(M, width) {
  const n = M.length;
  const parts = [];
  for (let off = 0; off < M[0].length; off += width) {
    const p = zeros(n, width);
    for (let i = 0; i < n; i++) for (let j = 0; j < width; j++) p[i][j] = M[i][off + j];
    parts.push(p);
  }
  return parts;
}

function negMatMul(A, B) {
  const C = matMul(A, B);
  for (const row of C) for (let j = 0; j < row.length; j++) row[j] = -row[j];
  return C;
}

/** Sensitivity of the model outputs to the model inputs (equations 24-33). */
function sensitivity(z, AP, An, AT, Am, invrat) {
  const nratios = An.length;
  const alpha = z[1], beta = z[2];

  const AN = new Float64Array(nratios), AM = new Float64Array(nratios);
  for (let i = 0; i < nratios; i++) {
    AN[i] = An[i] * Math.exp(-AP[i] * alpha);
    AM[i] = Am[i] * Math.exp(-AP[i] * beta);
  }

  const P = invrat.map((i) => AP[i]);
  const n = invrat.map((i) => An[i]);
  const T = invrat.map((i) => AT[i]);
  const m = invrat.map((i) => Am[i]);
  const dz = zSensitivity(z, P, n, T, m);
  if (!dz) return null;

  const full = (rows) => {
    const M = zeros(3, nratios);
    invrat.forEach((ix, j) => {
      M[0][ix] = rows.dzdn ? rows.dzdn[0][j] : 0;
      M[1][ix] = rows.dzdn ? rows.dzdn[1][j] : 0;
      M[2][ix] = rows.dzdn ? rows.dzdn[2][j] : 0;
    });
    return M;
  };
  const dzdAn = full({ dzdn: dz.dzdn });
  const dzdAm = full({ dzdn: dz.dzdm });
  const dzdAT = full({ dzdn: dz.dzdT });
  const dalphadAn = dzdAn[1], dalphadAm = dzdAm[1], dalphadAT = dzdAT[1];
  const dbetadAn = dzdAn[2], dbetadAm = dzdAm[2], dbetadAT = dzdAT[2];

  const outer = (v, w) => {
    const M = zeros(v.length, w.length);
    for (let i = 0; i < v.length; i++) for (let j = 0; j < w.length; j++) M[i][j] = v[i] * w[j];
    return M;
  };
  const negOuter = (v, w) => {
    const M = outer(v, w);
    for (const r of M) for (let j = 0; j < r.length; j++) r[j] = -r[j];
    return M;
  };
  const diagOf = (v) => {
    const M = zeros(v.length, v.length);
    for (let i = 0; i < v.length; i++) M[i][i] = v[i];
    return M;
  };

  const NP = new Float64Array(nratios), MP = new Float64Array(nratios);
  for (let i = 0; i < nratios; i++) { NP[i] = AN[i] * AP[i]; MP[i] = AM[i] * AP[i]; }

  const dANdAT = negOuter(NP, dalphadAT); // equation (26)
  const dANdAn = zeros(nratios, nratios);
  const eA = diagOf(AP.map((p) => Math.exp(-alpha * p)));
  const negNPAn = negOuter(NP, dalphadAn); // equation (24)
  for (let i = 0; i < nratios; i++)
    for (let j = 0; j < nratios; j++) dANdAn[i][j] = eA[i][j] + negNPAn[i][j];
  const dANdAm = negOuter(NP, dalphadAm); // equation (25)

  const dAMdAT = negOuter(MP, dbetadAT); // equation (33)
  const dAMdAn = negOuter(MP, dbetadAn); // equation (31)
  const dAMdAm = zeros(nratios, nratios);
  const eB = diagOf(AP.map((p) => Math.exp(-beta * p)));
  const negMPAm = negOuter(MP, dbetadAm); // equation (32)
  for (let i = 0; i < nratios; i++)
    for (let j = 0; j < nratios; j++) dAMdAm[i][j] = eB[i][j] + negMPAm[i][j];

  return { dzdAn, dzdAT, dzdAm, dANdAn, dANdAT, dANdAm, dAMdAn, dAMdAT, dAMdAm };
}

/** Equations (16), (23) and (30): the covariance matrices of z, AN and AM. */
function propagate(z, AP, An, AT, Am, VAn, VAT, VAm, invrat) {
  const s = sensitivity(z, AP, An, AT, Am, invrat);
  if (!s) return { Vz: zeros(3, 3), VAN: zeros(An.length, An.length), ok: false };
  const Vz = matAdd(matAdd(sandwich(s.dzdAn, VAn), sandwich(s.dzdAT, VAT)), sandwich(s.dzdAm, VAm));
  const VAN = matAdd(matAdd(sandwich(s.dANdAn, VAn), sandwich(s.dANdAT, VAT)), sandwich(s.dANdAm, VAm));
  const VAM = matAdd(matAdd(sandwich(s.dAMdAn, VAn), sandwich(s.dAMdAT, VAT)), sandwich(s.dAMdAm, VAm));
  return { Vz, VAN, VAM, ok: true };
}

/* ========================================================================== *
 * 5. the double spike inversion
 * ========================================================================== */

/**
 * Solve the double spike equations for one measurement by eliminating lambda
 * and running a damped Newton iteration on the remaining two unknowns
 * (alpha, beta).  This mirrors doublespike/inversion.py exactly.
 *
 *   lambda T_i + (1 - lambda) N_i = M_i                    equation (10)
 *   N_i = n_i exp(-alpha P_i),  M_i = m_i exp(-beta P_i)
 *
 * Because the equations are linear in lambda, for a trial (alpha, beta) we have
 * lambda = (M_i - N_i) / (T_i - N_i).  Requiring the pivot ratio i0 and a second
 * ratio x to agree gives the 2 x 2 system
 *
 *   G_x = S_i0 U_x - U_i0 S_x = 0,   S = T - N, U = M - N.
 *
 * Returns {alpha, beta, lambda, ok, residual, iterations} or null.
 */
function invert(P, n, T, m, opts) {
  const tol = (opts && opts.tol) || 1e-14;
  const maxiter = (opts && opts.maxiter) || 60;
  const maxstep = (opts && opts.maxstep) || 2.0;
  const nr = m.length;

  /* pivot: the ratio where the spike differs most from the standard */
  let i0 = 0, bestD = -1;
  for (let i = 0; i < nr; i++) {
    const d = Math.abs(T[i] - n[i]);
    if (d > bestD) { bestD = d; i0 = i; }
  }
  const others = [];
  for (let i = 0; i < nr; i++) if (i !== i0) others.push(i);
  // keep only the two most sensitive of the remaining ratios
  others.sort((a, b) => Math.abs(T[b] - n[b]) - Math.abs(T[a] - n[a]));
  const o = [others[0], others[1]];

  const S = (a, b, arr) => {
    // S = T - N, U = M - N at (a, b)
    const NS = new Float64Array(nr), US = new Float64Array(nr);
    for (let i = 0; i < nr; i++) {
      const N = n[i] * Math.exp(-a * P[i]);
      const M = m[i] * Math.exp(-b * P[i]);
      NS[i] = N; US[i] = M - N;
    }
    return { NS, US };
  };

  const residual = (a, b) => {
    const { NS, US } = S(a, b);
    const S0 = T[i0] - NS[i0], U0 = US[i0];
    const Sx = [T[o[0]] - NS[o[0]], T[o[1]] - NS[o[1]]];
    const Ux = [US[o[0]], US[o[1]]];
    const G = [S0 * Ux[0] - U0 * Sx[0], S0 * Ux[1] - U0 * Sx[1]];
    const jac = zeros(2, 2);
    const N0 = NS[i0], M0 = US[i0] + NS[i0];
    for (let k = 0; k < 2; k++) {
      const x = o[k];
      const Nx = NS[x], Mx = US[x] + NS[x], Sxv = T[x] - Nx, Uxv = US[x];
      jac[k][0] = P[i0] * N0 * Uxv + S0 * P[x] * Nx - P[x] * Nx * U0 - Sxv * P[i0] * N0;
      jac[k][1] = -S0 * P[x] * Mx + Sxv * P[i0] * M0;
    }
    return { G, jac, S0, U0 };
  };

  /* linear initial guess, equations (11)-(14).  The system is square only for
     the three ratios actually used (the pivot plus the two most sensitive
     others), so those are the only rows fed to the solver -- for a 5 isotope
     system nr is 4 and using all of them would make A non-square. */
  const rows = [i0, o[0], o[1]];
  const A = zeros(3, 3);
  const BB = zeros(3, 1);
  for (let k = 0; k < 3; k++) {
    const i = rows[k];
    A[k][0] = T[i] - n[i];
    A[k][1] = -n[i] * P[i];
    A[k][2] = m[i] * P[i];
    BB[k][0] = m[i] - n[i];
  }
  const y0 = solve(A, BB);
  let alpha = 0, beta = 0;
  if (y0) {
    const lam0 = y0[0][0];
    const cap = 1 / Math.max(...P.map(Math.abs));
    const a0 = Math.abs(1 - lam0) > 1e-12 ? y0[1][0] / (1 - lam0) : 0;
    const b0 = y0[2][0];
    if (
      isFinite(a0) && isFinite(b0) &&
      Math.abs(a0) <= cap && Math.abs(b0) <= cap && lam0 > -0.5 && lam0 < 1.5
    ) {
      alpha = a0; beta = b0;
    }
  }

  let iterations = 0, ok = false;
  for (let it = 0; it < maxiter; it++) {
    const r = residual(alpha, beta);
    const merit = Math.max(Math.abs(r.G[0]), Math.abs(r.G[1]));
    if (!isFinite(merit)) break;
    const step = solve(r.jac, [[-r.G[0]], [-r.G[1]]]);
    if (!step) break;
    let da = step[0][0], db = step[1][0];
    if (!isFinite(da) || !isFinite(db)) break;
    // direction preserving step limiting
    const snorm = Math.max(Math.abs(da), Math.abs(db));
    if (snorm > maxstep) { const f = maxstep / snorm; da *= f; db *= f; }

    let t = 1, accepted = false;
    for (let ls = 0; ls < 50; ls++) {
      const ra = residual(alpha + t * da, beta + t * db);
      const m2 = Math.max(Math.abs(ra.G[0]), Math.abs(ra.G[1]));
      if (m2 <= (1 - 1e-4 * t) * merit) { accepted = true; break; }
      t *= 0.5;
    }
    if (!accepted) break;
    alpha += t * da;
    beta += t * db;
    iterations = it + 1;
    if (Math.max(Math.abs(t * da), Math.abs(t * db)) < 1e-15) { ok = true; break; }
  }

  const rf = residual(alpha, beta);
  const lambda = rf.U0 / rf.S0;
  // verify on the original equations
  let maxres = 0;
  for (let i = 0; i < nr; i++) {
    const N = n[i] * Math.exp(-alpha * P[i]);
    const M = m[i] * Math.exp(-beta * P[i]);
    const lhs = lambda * T[i] + (1 - lambda) * N;
    const den = Math.abs(lambda * T[i]) + Math.abs((1 - lambda) * N) + Math.abs(M) + 1e-300;
    maxres = Math.max(maxres, Math.abs(lhs - M) / den);
  }
  return {
    alpha, beta, lambda, residual: maxres, iterations,
    ok: ok && isFinite(maxres) && maxres < 1e-8,
  };
}

/** Full data reduction for a matrix of measured beam intensities.
 *
 * The inversion itself only uses the four isotopes chosen for it (giving three
 * ratios), exactly like the Python library; the sample and mixture
 * compositions are then reconstructed over the full isotope set.
 */
function reduce(sys, measured, spike, isoinvIndices, standard) {
  const nisos = sys.isonum.length;
  const spikeN = normalise(spike);

  // denominator = the isotope the spike contains most of; the library only
  // reorders when the first inversion isotope has essentially no spike
  const first = isoinvIndices[0];
  let order = isoinvIndices.slice();
  if (spikeN[first] < 0.001) {
    let deno = first, best = -1;
    for (const ix of isoinvIndices) if (spikeN[ix] > best) { best = spikeN[ix]; deno = ix; }
    order = [deno].concat(isoinvIndices.filter((ix) => ix !== deno));
  }
  const deno = order[0];

  /* ---- three ratios used by the inversion ---- */
  const k = order.length;
  const P = new Float64Array(k - 1), n = new Float64Array(k - 1);
  const T = new Float64Array(k - 1), m = new Float64Array(k - 1);
  for (let i = 1; i < k; i++) {
    P[i - 1] = Math.log(sys.mass[order[i]] / sys.mass[deno]);
    n[i - 1] = standard[order[i]] / standard[deno];
    T[i - 1] = spikeN[order[i]] / spikeN[deno];
    m[i - 1] = measured[order[i]] / measured[deno];
  }
  const res = invert(P, n, T, m);
  if (!res) return null;

  /* ---- full ratio sets, used to rebuild the compositions ---- */
  const allNi = ratioIndices(deno, nisos);
  const APfull = new Float64Array(allNi.length);
  const AnFull = new Float64Array(allNi.length);
  const ATfull = new Float64Array(allNi.length);
  const AmFull = new Float64Array(allNi.length);
  for (let i = 0; i < allNi.length; i++) {
    APfull[i] = Math.log(sys.mass[allNi[i]] / sys.mass[deno]);
    AnFull[i] = standard[allNi[i]] / standard[deno];
    ATfull[i] = spikeN[allNi[i]] / spikeN[deno];
    AmFull[i] = measured[allNi[i]] / measured[deno];
  }
  const ANfull = new Float64Array(allNi.length);
  const AMfull = new Float64Array(allNi.length);
  const sample = new Float64Array(nisos).fill(1);
  const mixture = new Float64Array(nisos).fill(1);
  for (let i = 0; i < allNi.length; i++) {
    ANfull[i] = AnFull[i] * Math.exp(-res.alpha * APfull[i]);
    AMfull[i] = AmFull[i] * Math.exp(-res.beta * APfull[i]);
    sample[allNi[i]] = ANfull[i];
    mixture[allNi[i]] = AMfull[i];
  }

  return {
    ...res,
    prop: ratioPropToRealProp(res.lambda, ATfull, ANfull),
    sample: normalise(sample),
    mixture: normalise(mixture),
  };
}

/* ========================================================================== *
 * 6. optimal double spike search
 * ========================================================================== */

/**
 * Search for the (spike-sample proportion, spike composition) that minimises the
 * error, using a coarse grid followed by an adaptive per axis pattern search and
 * then a fine pattern search.  Fully deterministic and derivative free.
 */
function optimalSpike(sys, prep, emod, spike1, spike2, alpha, beta, gridN) {
  const n = gridN || 9;
  const P_LO = 1e-3, P_HI = 1 - 1e-3;

  const err = (p, q) => {
    const sp = mix(spike1, spike2, q);
    const e = errorEstimate(sys, prep, emod, p, sp, alpha, beta).error;
    return isFinite(e) ? e : Infinity;
  };

  let best = { p: 0.5, q: 0.5, e: Infinity };
  for (let i = 0; i < n; i++) {
    const p = P_LO + ((P_HI - P_LO) * i) / (n - 1);
    for (let j = 0; j < n; j++) {
      const q = P_LO + ((P_HI - P_LO) * j) / (n - 1);
      const e = err(p, q);
      if (e < best.e) best = { p, q, e };
    }
  }

  const stencil = [-1, 0, 1];
  let hp = 0.15, hq = 0.15;
  for (let it = 0; it < 400 && Math.max(hp, hq) > 1e-6; it++) {
    let improved = false;
    for (const a of stencil) {
      for (const b of stencil) {
        const p = Math.min(P_HI, Math.max(P_LO, best.p + a * hp));
        const q = Math.min(P_HI, Math.max(P_LO, best.q + b * hq));
        const e = err(p, q);
        if (e < best.e * (1 - 1e-13)) {
          const movedP = a !== 0, movedQ = b !== 0;
          best = { p, q, e };
          hp = movedP ? Math.min(hp * 1.7, 0.4) : hp * 0.5;
          hq = movedQ ? Math.min(hq * 1.7, 0.4) : hq * 0.5;
          improved = true;
        }
      }
    }
    if (!improved) { hp *= 0.5; hq *= 0.5; }
  }
  return best;
}

function mix(s1, s2, q) {
  const out = new Float64Array(s1.length);
  for (let i = 0; i < s1.length; i++) out[i] = q * s1[i] + (1 - q) * s2[i];
  return normalise(out);
}

/* ========================================================================== *
 * 7. Monte Carlo
 * ========================================================================== */

/** Box-Muller normal deviate. */
function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Deterministic PRNG so that a seed reproduces a run exactly. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Simulate a mass spectrometer run, mirroring monte.monterun. */
function monteRun(sys, emod, prop, spike, alpha, beta, n, seed) {
  const rng = mulberry32(seed === undefined ? 12345 : seed);
  const nisos = sys.isonum.length;
  const spikeN = normalise(spike);
  const sample = normalise(sys.standard.map((s, i) => s * Math.exp(-Math.log(sys.mass[i]) * alpha)));
  const mixture = new Float64Array(nisos);
  for (let i = 0; i < nisos; i++) mixture[i] = prop * spikeN[i] + (1 - prop) * sample[i];
  const measured = normalise(mixture.map((v, i) => v * Math.exp(Math.log(sys.mass[i]) * beta)));

  const out = [];
  for (let r = 0; r < n; r++) {
    const beam = new Float64Array(nisos);
    for (let i = 0; i < nisos; i++) {
      const mean = emod.measured.intensity * measured[i];
      const varr = emod.measured.a[i] + emod.measured.b[i] * mean + emod.measured.c[i] * mean * mean;
      beam[i] = mean + Math.sqrt(Math.max(varr, 0)) * gauss(rng);
    }
    out.push(beam);
  }
  return out;
}

/* ========================================================================== *
 * 8. exports (attached to window for the single file build)
 * ========================================================================== */

const DS = {
  zeros, identity, matMul, matVec, transpose, sandwich, matAdd, solve,
  normalise, sum, ratiosOf, ratioIndices, ratioIndex, compositionOf,
  ratioPropToRealProp, realPropToRatioProp,
  buildErrorModel, calcRatioCov, covBeamToRatio,
  prepareSystem, errorEstimate, changeDenomCov, sensitivity, propagate, zSensitivity,
  invert, reduce, optimalSpike, mix, monteRun, mulberry32, gauss,
  isotopeIndices, indexOfIsotope,
};

if (typeof module !== 'undefined' && module.exports) module.exports = DS;
if (typeof window !== 'undefined') window.DS = DS;
