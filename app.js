"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  dim: 3,
  samples: 12,
  selectedNode: null,
  arch: null,
  task: null,
  model: null,
  trainedModel: null,
  theory: null,
  currentRun: null,
  overlays: [],
  gallery: [],
  training: false
};

const palette = ["#48d6a3", "#f7c95f", "#66a7ff", "#ff756d", "#c49bff", "#7bdff2"];

class RNG {
  constructor(seed) { this.s = (seed >>> 0) || 1; }
  next() {
    this.s ^= this.s << 13; this.s >>>= 0;
    this.s ^= this.s >>> 17; this.s >>>= 0;
    this.s ^= this.s << 5; this.s >>>= 0;
    return (this.s >>> 0) / 4294967296;
  }
  normal() {
    const u = Math.max(this.next(), 1e-12), v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}

function cfg() {
  const alpha = Math.pow(10, Number($("alpha").value));
  const lr = Math.pow(10, Number($("lr").value));
  return {
    depth: Math.max(0, Number($("depth").value)),
    width: Number($("width").value),
    ioDim: Math.max(1, Number($("ioDim").value)),
    alpha,
    modelSeed: Number($("modelSeed").value),
    taskSeed: Number($("taskSeed").value),
    optimizer: $("optimizer").value,
    lr,
    steps: Math.max(1, Number($("steps").value)),
    beta1: Number($("beta1").value),
    beta2: Number($("beta2").value),
    epsilon: Number($("epsilon").value),
    weightDecay: Number($("weightDecay").value)
  };
}

function fmt(x) {
  if (!Number.isFinite(x)) return "-";
  const ax = Math.abs(x);
  if (ax === 0) return "0";
  if (ax < 1e-3 || ax >= 1e4) return x.toExponential(3);
  return x.toPrecision(5);
}

function fmtTick(x) {
  if (!Number.isFinite(x)) return "";
  const ax = Math.abs(x);
  if (ax === 0) return "0";
  if (ax < 1e-2 || ax >= 1e5) return x.toExponential(1);
  if (ax < 1) return x.toPrecision(2);
  return String(Number(x.toFixed(2)));
}

function linearTicks(min, max, count) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [0, 1];
  const step = (max - min) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, i) => min + i * step);
}

function logTicks(min, max) {
  const lo = Math.floor(Math.log10(Math.max(min, 1e-14)));
  const hi = Math.ceil(Math.log10(Math.max(max, 1e-14)));
  const ticks = [];
  const stride = Math.max(1, Math.ceil((hi - lo) / 7));
  for (let p = lo; p <= hi; p += stride) ticks.push(Math.pow(10, p));
  if (ticks[ticks.length - 1] < Math.pow(10, hi)) ticks.push(Math.pow(10, hi));
  return ticks.length ? ticks : [min, max];
}

function setStatus(text) { $("status").textContent = text; }

function makeTask(seed) {
  const rng = new RNG(seed), X = [], Y = [], A = [];
  for (let r = 0; r < state.dim; r++) {
    A[r] = [];
    for (let c = 0; c < state.dim; c++) A[r][c] = rng.normal() / Math.sqrt(state.dim);
  }
  for (let i = 0; i < state.samples; i++) {
    const x = Array.from({ length: state.dim }, () => rng.normal());
    X.push(x);
    Y.push(matVec(A, x));
  }
  return { X, Y, A };
}

function makeArch(depth, width, previousEdges) {
  const dims = [state.dim];
  for (let i = 0; i < depth; i++) dims.push(width);
  dims.push(state.dim);
  const last = dims.length - 1;
  const edges = [];
  for (let i = 0; i < last; i++) edges.push({ from: i, to: i + 1, chain: true });
  if (previousEdges) {
    for (const e of previousEdges) {
      const compatible = e.to <= last && e.from < e.to && dims[e.from] === dims[e.to];
      if (compatible && !edges.some((x) => !x.chain && x.from === e.from && x.to === e.to)) {
        edges.push({ from: e.from, to: e.to, chain: false });
      }
    }
  }
  return { depth, width, dims, edges };
}

function initModel(arch, seed, alpha) {
  const rng = new RNG(seed);
  return {
    edges: arch.edges.map((e) => {
      const rows = arch.dims[e.to], cols = arch.dims[e.from];
      if (!e.chain) return { ...e, rows, cols };
      const w = new Float64Array(rows * cols);
      const scale = alpha / Math.sqrt(cols);
      for (let i = 0; i < w.length; i++) w[i] = rng.normal() * scale;
      return { ...e, rows, cols, w, v: new Float64Array(w.length), m: new Float64Array(w.length), t: 0 };
    })
  };
}

function initModelPreservingWeights(arch, seed, alpha, previousModel) {
  const next = initModel(arch, seed, alpha);
  if (!previousModel) return next;
  for (const edge of next.edges) {
    if (!edge.chain || !edge.w) continue;
    const prev = previousModel.edges.find((e) => e.chain && e.from === edge.from && e.to === edge.to && e.w?.length === edge.w.length);
    if (!prev) continue;
    edge.w.set(prev.w);
    if (prev.v?.length === edge.v.length) edge.v.set(prev.v);
    if (prev.m?.length === edge.m.length) edge.m.set(prev.m);
    edge.t = prev.t || 0;
  }
  return next;
}

function matVec(A, x) {
  return A.map((row) => row.reduce((s, a, i) => s + a * x[i], 0));
}

function forward(model, arch, x) {
  const acts = arch.dims.map((d) => new Float64Array(d));
  acts[0].set(x);
  for (let n = 1; n < arch.dims.length; n++) {
    for (const e of model.edges) {
      if (e.to !== n) continue;
      const src = acts[e.from], dst = acts[n];
      if (!e.chain) {
        for (let k = 0; k < dst.length; k++) dst[k] += src[k];
        continue;
      }
      for (let r = 0; r < e.rows; r++) {
        let sum = 0;
        const off = r * e.cols;
        for (let c = 0; c < e.cols; c++) sum += e.w[off + c] * src[c];
        dst[r] += sum;
      }
    }
  }
  return acts;
}

function lossAndGrad(model, arch, task) {
  const grads = model.edges.map((e) => e.chain ? new Float64Array(e.w.length) : null);
  let loss = 0;
  const norm = task.X.length * state.dim;
  const layerGradNorms = model.edges.map(() => 0);
  const residuals = [];
  for (let s = 0; s < task.X.length; s++) {
    const acts = forward(model, arch, task.X[s]);
    const out = acts[acts.length - 1];
    const deltas = arch.dims.map((d) => new Float64Array(d));
    const r = new Float64Array(state.dim);
    for (let i = 0; i < state.dim; i++) {
      r[i] = out[i] - task.Y[s][i];
      deltas[deltas.length - 1][i] = r[i] / norm;
      loss += 0.5 * r[i] * r[i] / norm;
    }
    residuals.push(Array.from(r));
    for (let n = arch.dims.length - 1; n >= 1; n--) {
      for (let ei = model.edges.length - 1; ei >= 0; ei--) {
        const e = model.edges[ei];
        if (e.to !== n) continue;
        const d = deltas[e.to], a = acts[e.from], g = grads[ei], up = deltas[e.from];
        if (!e.chain) {
          for (let k = 0; k < up.length; k++) up[k] += d[k];
          continue;
        }
        for (let rix = 0; rix < e.rows; rix++) {
          const doff = rix * e.cols;
          for (let c = 0; c < e.cols; c++) {
            g[doff + c] += d[rix] * a[c];
            up[c] += e.w[doff + c] * d[rix];
          }
        }
      }
    }
  }
  let gradNorm2 = 0;
  for (let i = 0; i < grads.length; i++) {
    let n2 = 0;
    if (!grads[i]) {
      layerGradNorms[i] = 0;
      continue;
    }
    for (const v of grads[i]) n2 += v * v;
    layerGradNorms[i] = Math.sqrt(n2);
    gradNorm2 += n2;
  }
  return { loss, grads, gradNorm2, gradNorm: Math.sqrt(gradNorm2), residuals, layerGradNorms };
}

function applyStep(model, grads, c) {
  const opt = c.optimizer;
  for (let ei = 0; ei < model.edges.length; ei++) {
    const e = model.edges[ei], g = grads[ei];
    if (!e.chain) continue;
    e.t += 1;
    for (let i = 0; i < e.w.length; i++) {
      let gi = g[i];
      if (opt === "SGD") {
        gi += c.weightDecay * e.w[i];
        e.v[i] = c.beta1 * e.v[i] + gi;
        e.w[i] -= c.lr * e.v[i];
      } else {
        if (opt === "Adam") gi += c.weightDecay * e.w[i];
        e.m[i] = c.beta1 * e.m[i] + (1 - c.beta1) * gi;
        e.v[i] = c.beta2 * e.v[i] + (1 - c.beta2) * gi * gi;
        const mh = e.m[i] / (1 - Math.pow(c.beta1, e.t));
        const vh = e.v[i] / (1 - Math.pow(c.beta2, e.t));
        if (opt === "AdamW") e.w[i] *= (1 - c.lr * c.weightDecay);
        e.w[i] -= c.lr * mh / (Math.sqrt(vh) + c.epsilon);
      }
    }
  }
}

function cloneModel(model) {
  return {
    edges: model.edges.map((e) => ({
      ...e,
      w: e.w ? new Float64Array(e.w) : undefined,
      v: e.v ? new Float64Array(e.v) : undefined,
      m: e.m ? new Float64Array(e.m) : undefined,
      t: e.t
    }))
  };
}

function paramCount(model) {
  return model.edges.reduce((s, e) => s + (e.w?.length || 0), 0);
}

function gradientVectorForOutput(model, arch, x, outIndex) {
  const acts = forward(model, arch, x);
  const deltas = arch.dims.map((d) => new Float64Array(d));
  deltas[deltas.length - 1][outIndex] = 1;
  const vec = [];
  for (let n = arch.dims.length - 1; n >= 1; n--) {
    for (let ei = model.edges.length - 1; ei >= 0; ei--) {
      const e = model.edges[ei];
      if (e.to !== n) continue;
      const d = deltas[e.to], a = acts[e.from], up = deltas[e.from];
      if (!e.chain) {
        for (let k = 0; k < up.length; k++) up[k] += d[k];
        vec[ei] = null;
        continue;
      }
      const g = new Float64Array(e.w.length);
      for (let r = 0; r < e.rows; r++) {
        const off = r * e.cols;
        for (let c = 0; c < e.cols; c++) {
          g[off + c] = d[r] * a[c];
          up[c] += e.w[off + c] * d[r];
        }
      }
      vec[ei] = g;
    }
  }
  return vec;
}

function computeTheory(model, arch, task, base) {
  const rows = task.X.length * state.dim;
  const params = paramCount(model);
  const ops = rows * rows * params;
  const theory = {
    l0: base.loss,
    gradNorm2: base.gradNorm2,
    gradNorm: base.gradNorm,
    slope: -base.gradNorm2,
    ntkSlope: NaN,
    ntkPrediction: [],
    ntkSkipped: ops > 100000000
  };
  if (theory.ntkSkipped) return theory;

  const J = [];
  const residual = new Float64Array(rows);
  let row = 0;
  for (let s = 0; s < task.X.length; s++) {
    for (let k = 0; k < state.dim; k++) {
      J[row] = gradientVectorForOutput(model, arch, task.X[s], k);
      residual[row] = base.residuals[s][k];
      row++;
    }
  }
  const K = Array.from({ length: rows }, () => new Float64Array(rows));
  for (let a = 0; a < rows; a++) {
    for (let b = a; b < rows; b++) {
      let dot = 0;
      for (let ei = 0; ei < model.edges.length; ei++) {
        const ga = J[a][ei], gb = J[b][ei];
        if (!ga || !gb) continue;
        for (let p = 0; p < ga.length; p++) dot += ga[p] * gb[p];
      }
      K[a][b] = dot;
      K[b][a] = dot;
    }
  }
  let rKr = 0;
  for (let i = 0; i < rows; i++) for (let j = 0; j < rows; j++) rKr += residual[i] * K[i][j] * residual[j];
  const norm = task.X.length * state.dim;
  theory.ntkSlope = -rKr / (norm * norm);
  theory.sanity = Math.abs((-theory.ntkSlope) - theory.gradNorm2) / Math.max(1e-12, theory.gradNorm2);
  theory.K = K;
  theory.r0 = residual;
  return theory;
}

function ntkTrajectory(theory, c) {
  if (!theory.K) return [];
  const n = state.samples, d = state.dim, dt = c.lr / (n * d);
  let r = new Float64Array(theory.r0);
  const out = [{ step: 0, t: 0, loss: theory.l0 }];
  for (let step = 1; step <= c.steps; step++) {
    const next = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) {
      let kr = 0;
      for (let j = 0; j < r.length; j++) kr += theory.K[i][j] * r[j];
      next[i] = r[i] - dt * kr;
    }
    r = next;
    if (step % Math.max(1, Math.floor(c.steps / 240)) === 0 || step === c.steps) {
      let l = 0;
      for (const v of r) l += v * v;
      out.push({ step, t: step * c.lr, loss: 0.5 * l / (n * d) });
    }
  }
  return out;
}

function effectiveMap(model, arch) {
  let mats = arch.dims.map((d, i) => {
    const m = Array.from({ length: d }, () => new Float64Array(state.dim));
    if (i === 0) for (let k = 0; k < state.dim; k++) m[k][k] = 1;
    return m;
  });
  for (let n = 1; n < arch.dims.length; n++) {
    for (const e of model.edges) {
      if (e.to !== n) continue;
      if (!e.chain) {
        for (let r = 0; r < mats[n].length; r++) {
          for (let k = 0; k < state.dim; k++) mats[n][r][k] += mats[e.from][r][k];
        }
        continue;
      }
      for (let r = 0; r < e.rows; r++) {
        const off = r * e.cols;
        for (let c = 0; c < e.cols; c++) {
          for (let k = 0; k < state.dim; k++) mats[n][r][k] += e.w[off + c] * mats[e.from][c][k];
        }
      }
    }
  }
  return mats[mats.length - 1];
}

function mapDistance(M, A) {
  let s = 0;
  for (let r = 0; r < A.length; r++) for (let c = 0; c < A[0].length; c++) s += (M[r][c] - A[r][c]) ** 2;
  return Math.sqrt(s);
}

function deviation(actual, ntk) {
  if (!actual.length || !ntk.length) return NaN;
  let num = 0, den = 0, j = 0;
  for (let i = 1; i < actual.length; i++) {
    while (j < ntk.length - 1 && ntk[j + 1].step <= actual[i].step) j++;
    const a0 = actual[i - 1], a1 = actual[i], p = ntk[j] || ntk[ntk.length - 1];
    const dx = a1.step - a0.step;
    num += Math.abs(a1.loss - p.loss) * dx;
    den += Math.abs(p.loss) * dx;
  }
  return den > 0 ? num / den : NaN;
}

function rebuild(resetEdges = true) {
  const c = cfg();
  state.dim = c.ioDim;
  $("depthValue").textContent = c.depth;
  $("alphaValue").textContent = c.alpha.toPrecision(3);
  $("lrValue").textContent = c.lr.toPrecision(3);
  const previous = resetEdges || !state.arch ? null : state.arch.edges.filter((e) => !e.chain);
  state.arch = makeArch(c.depth, c.width, previous);
  state.task = makeTask(c.taskSeed);
  state.model = initModel(state.arch, c.modelSeed, c.alpha);
  state.trainedModel = null;
  state.selectedNode = null;
  state.currentRun = null;
  recompute();
}

function recompute() {
  const c = cfg();
  const base = lossAndGrad(state.model, state.arch, state.task);
  state.theory = computeTheory(state.model, state.arch, state.task, base);
  state.trainedModel = null;
  updateDiagnostics(base.loss, NaN);
  drawGraph();
  drawPlot();
  setStatus(state.theory.ntkSkipped ? "Theory ready, NTK curve skipped for size" : "Theory ready");
}

async function train() {
  if (state.training) return;
  state.training = true;
  setStatus("Training...");
  const c = cfg();
  const model = cloneModel(state.model);
  const actual = [];
  const every = Math.max(1, Math.floor(c.steps / 240));
  let firstStepDelta = NaN;
  const l0 = lossAndGrad(model, state.arch, state.task).loss;
  actual.push({ step: 0, t: 0, loss: l0 });
  for (let step = 1; step <= c.steps; step++) {
    const before = step === 1 ? lossAndGrad(model, state.arch, state.task).loss : null;
    const lg = lossAndGrad(model, state.arch, state.task);
    applyStep(model, lg.grads, c);
    if (step === 1) firstStepDelta = (lossAndGrad(model, state.arch, state.task).loss - before) / c.lr;
    if (step % every === 0 || step === c.steps) {
      actual.push({ step, t: step * c.lr, loss: lossAndGrad(model, state.arch, state.task).loss });
      state.currentRun = { actual, ntk: state.theory.ntkPrediction || [], color: palette[0], maxStep: c.steps };
      state.trainedModel = model;
      drawPlot();
      updateDiagnostics(actual[actual.length - 1].loss, NaN);
      await new Promise(requestAnimationFrame);
    }
  }
  const ntk = ntkTrajectory(state.theory, c);
  const dev = deviation(actual, ntk);
  state.trainedModel = model;
  state.theory.ntkPrediction = ntk;
  const finalLoss = actual[actual.length - 1].loss;
  const card = makeExperiment(c, actual, ntk, dev, firstStepDelta, model);
  state.gallery.unshift(card);
  saveGallery();
  state.currentRun = { actual, ntk, color: palette[0], maxStep: c.steps };
  updateDiagnostics(finalLoss, dev);
  renderGallery();
  drawGraph();
  drawPlot();
  state.training = false;
  setStatus("Run saved");
}

function makeExperiment(c, actual, ntk, dev, firstStepDelta, model) {
  const l0 = state.theory.l0;
  const finalLoss = actual[actual.length - 1].loss;
  const threshold = l0 * 0.1;
  const conv = (actual.find((p) => p.loss <= threshold) || {}).step ?? Infinity;
  return {
    id: Date.now(),
    name: "#" + String(state.gallery.length + 1).padStart(2, "0"),
    created: Date.now(),
    config: c,
    arch: JSON.parse(JSON.stringify(state.arch)),
    diagnostics: {
      l0,
      slope: state.theory.slope,
      gradNorm2: state.theory.gradNorm2,
      ntkSlope: state.theory.ntkSlope,
      ntkDeviation: dev,
      firstStepDelta,
      finalLoss,
      conv,
      mapDistance: mapDistance(effectiveMap(model, state.arch), state.task.A)
    },
    actual,
    ntk,
    weights: model.edges.map((e) => e.w ? Array.from(e.w) : null)
  };
}

function updateDiagnostics(finalLoss, dev) {
  const th = state.theory || {};
  $("diagL0").textContent = fmt(th.l0);
  $("diagSlope").textContent = fmt(th.slope);
  $("diagGrad").textContent = fmt(th.gradNorm2);
  $("diagNtk").textContent = Number.isFinite(th.ntkSlope) ? fmt(th.ntkSlope) : "skipped";
  $("diagDev").textContent = Number.isFinite(dev) ? fmt(dev) : "-";
  $("diagFinal").textContent = Number.isFinite(finalLoss) ? fmt(finalLoss) : "-";
  const mapModel = state.trainedModel || state.model;
  $("diagMap").textContent = mapModel ? fmt(mapDistance(effectiveMap(mapModel, state.arch), state.task.A)) : "-";
}

function drawGraph() {
  const svg = $("graph");
  svg.innerHTML = "";
  const count = state.arch.dims.length;
  const pts = state.arch.dims.map((_, i) => ({
    x: 70 + i * (760 / Math.max(1, count - 1)),
    y: 170
  }));
  const weights = [];
  for (let i = 0; i < count - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    weights[i] = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }
  for (const e of state.arch.edges) {
    const a = pts[e.from], b = pts[e.to];
    if (e.chain) {
      const w = weights[e.from];
      const left = document.createElementNS("http://www.w3.org/2000/svg", "line");
      left.setAttribute("x1", a.x + 25);
      left.setAttribute("y1", a.y);
      left.setAttribute("x2", w.x - 24);
      left.setAttribute("y2", w.y);
      left.setAttribute("class", "edge chain");
      const right = document.createElementNS("http://www.w3.org/2000/svg", "line");
      right.setAttribute("x1", w.x + 24);
      right.setAttribute("y1", w.y);
      right.setAttribute("x2", b.x - 25);
      right.setAttribute("y2", b.y);
      right.setAttribute("class", "edge chain");
      svg.append(left, right);
    } else {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const bend = Math.min(120, Math.abs(b.x - a.x) * 0.32);
      const lift = 70 + 12 * (e.to - e.from);
      const d = `M ${a.x} ${a.y - 24} C ${a.x + bend} ${a.y - lift}, ${b.x - bend} ${b.y - lift}, ${b.x} ${b.y - 24}`;
      const removeSkip = (ev) => {
        ev.stopPropagation();
        state.arch.edges = state.arch.edges.filter((x) => x.chain || x.from !== e.from || x.to !== e.to);
        state.model = initModelPreservingWeights(state.arch, cfg().modelSeed, cfg().alpha, state.model);
        state.trainedModel = null;
        recompute();
      };
      path.setAttribute("d", d);
      path.setAttribute("class", "edge skip-edge");
      path.addEventListener("click", removeSkip);
      const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
      hit.setAttribute("d", d);
      hit.setAttribute("class", "skip-hit");
      hit.addEventListener("click", removeSkip);
      svg.append(path, hit);
    }
  }
  for (let i = 0; i < count - 1; i++) {
    const w = weights[i];
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "weight-matrix");
    g.setAttribute("transform", `translate(${w.x},${w.y})`);
    const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    box.setAttribute("x", -24);
    box.setAttribute("y", -19);
    box.setAttribute("width", 48);
    box.setAttribute("height", 38);
    box.setAttribute("rx", 4);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.textContent = `W${i + 1}`;
    const cap = document.createElementNS("http://www.w3.org/2000/svg", "text");
    cap.setAttribute("class", "caption");
    cap.setAttribute("y", 35);
    cap.textContent = `${state.arch.dims[i + 1]}x${state.arch.dims[i]}`;
    g.append(box, t, cap);
    svg.appendChild(g);
  }
  for (let i = 0; i < count; i++) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "node" + (state.selectedNode === i ? " selected" : ""));
    g.setAttribute("transform", `translate(${pts[i].x},${pts[i].y})`);
    g.addEventListener("click", () => clickNode(i));
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("r", i === 0 || i === count - 1 ? 30 : 24);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.textContent = i === 0 ? "x" : i === count - 1 ? "yhat" : `h${i}`;
    const cap = document.createElementNS("http://www.w3.org/2000/svg", "text");
    cap.setAttribute("class", "caption");
    cap.setAttribute("y", 48);
    cap.textContent = "d=" + state.arch.dims[i];
    g.append(c, t, cap);
    svg.appendChild(g);
  }
}

function clickNode(i) {
  if (state.selectedNode === null) {
    state.selectedNode = i;
  } else {
    const a = state.selectedNode, b = i;
    const compatible = state.arch.dims[a] === state.arch.dims[b];
    const hasSkip = state.arch.edges.some((e) => !e.chain && e.from === a && e.to === b);
    if (a < b && compatible && !hasSkip) {
      state.arch.edges.push({ from: a, to: b, chain: false });
      state.model = initModelPreservingWeights(state.arch, cfg().modelSeed, cfg().alpha, state.model);
      state.trainedModel = null;
      state.selectedNode = null;
      recompute();
      return;
    }
    if (a < b && !compatible) setStatus("Skip needs matching state dimensions");
    if (a < b && compatible && hasSkip) setStatus("Skip already exists");
    state.selectedNode = i === a ? null : i;
  }
  drawGraph();
}

function drawPlot() {
  const canvas = $("plot"), ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height, pad = { l: 86, r: 26, t: 30, b: 66 };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0b0f10";
  ctx.fillRect(0, 0, w, h);
  const series = [];
  if (state.currentRun) {
    series.push({ label: "Actual", pts: state.currentRun.actual, color: palette[0], dash: [] });
    if (state.currentRun.ntk?.length) series.push({ label: "Frozen NTK", pts: state.currentRun.ntk, color: palette[1], dash: [8, 7] });
  }
  state.overlays.forEach((e, i) => series.push({ label: e.name, pts: e.actual, color: palette[(i + 2) % palette.length], dash: [] }));
  const log = $("logAxis").checked;
  const logX = $("logXAxis").checked;
  const all = series.flatMap((s) => s.pts);
  const configuredSteps = cfg().steps;
  const runMaxStep = state.currentRun?.maxStep ?? configuredSteps;
  const overlayMaxStep = state.overlays.reduce((m, e) => Math.max(m, e.config?.steps || 0), 0);
  const maxX = Math.max(runMaxStep, overlayMaxStep, configuredSteps, 1);
  const ys = all.length ? all.map((p) => Math.max(p.loss, 1e-14)) : [0, 1];
  let minY = all.length ? Math.min(...ys) : 0;
  let maxY = all.length ? Math.max(...ys) : 1;
  if (log) {
    minY = Math.max(minY * 0.8, 1e-14);
    maxY = Math.max(maxY * 1.15, minY * 10);
  } else {
    minY = 0;
    maxY = maxY <= 0 ? 1 : maxY * 1.08;
  }
  const sx = (x) => {
    const v = logX ? Math.log1p(Math.max(0, x)) : x;
    const b = logX ? Math.log1p(maxX) : maxX;
    return pad.l + (v / b) * (w - pad.l - pad.r);
  };
  const sy = (y) => {
    const v = log ? Math.log10(Math.max(y, 1e-14)) : y;
    const a = log ? Math.log10(minY) : minY;
    const b = log ? Math.log10(maxY) : maxY;
    return h - pad.b - ((v - a) / (b - a)) * (h - pad.t - pad.b);
  };
  const xTicks = logX ? logTicks(1, maxX + 1).map((tick) => tick - 1).filter((tick) => tick >= 0 && tick <= maxX) : linearTicks(0, maxX, 6);
  const yTicks = log ? logTicks(minY, maxY) : linearTicks(minY, maxY, 6);

  ctx.strokeStyle = "#2d3638";
  ctx.lineWidth = 1;
  ctx.font = "12px ui-monospace";
  ctx.fillStyle = "#8f9d9a";
  ctx.textBaseline = "middle";
  yTicks.forEach((tick) => {
    if (tick < minY || tick > maxY) return;
    const y = sy(tick);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = "#8f9d9a";
    ctx.textAlign = "right";
    ctx.fillText(fmtTick(tick), pad.l - 10, y);
  });
  xTicks.forEach((tick) => {
    const x = sx(tick);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.fillStyle = "#8f9d9a";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(fmtTick(tick), x, h - pad.b + 12);
  });
  ctx.strokeStyle = "#607174";
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.fillStyle = "#cfe0dc";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(logX ? "log(step + 1)" : "step", pad.l + (w - pad.l - pad.r) / 2, h - 14);
  ctx.save();
  ctx.translate(20, pad.t + (h - pad.t - pad.b) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(log ? "loss L(t), log scale" : "loss L(t)", 0, 0);
  ctx.restore();
  if (!series.length) {
    ctx.fillStyle = "#8f9d9a";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = "16px ui-monospace";
    ctx.fillText("Train an experiment to draw L(t)", pad.l + 18, pad.t + (h - pad.t - pad.b) / 2);
    return;
  }
  ctx.font = "12px ui-monospace";
  series.forEach((s, i) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(s.dash);
    ctx.beginPath();
    s.pts.forEach((p, ix) => ix ? ctx.lineTo(sx(p.step), sy(p.loss)) : ctx.moveTo(sx(p.step), sy(p.loss)));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = s.color;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(s.label, pad.l + i * 145, 17);
  });
}

function saveGallery() {
  localStorage.setItem("linear-flow-gallery", JSON.stringify(state.gallery.slice(0, 48)));
}

function loadGallery() {
  try { state.gallery = JSON.parse(localStorage.getItem("linear-flow-gallery") || "[]"); }
  catch { state.gallery = []; }
  state.gallery.forEach(normalizeExperiment);
}

function normalizeExperiment(exp) {
  if (!exp.config) return exp;
  if (exp.config.alpha == null && exp.config.sigma != null) exp.config.alpha = exp.config.sigma;
  if (exp.config.ioDim == null) exp.config.ioDim = exp.arch?.dims?.[0] || state.dim;
  if (exp.arch?.dims && Array.isArray(exp.arch.edges)) {
    const chainKeys = new Set();
    for (let i = 0; i < exp.arch.dims.length - 1; i++) chainKeys.add(`${i}:${i + 1}`);
    exp.arch.edges = exp.arch.edges
      .filter((e) => e.from < e.to && e.to < exp.arch.dims.length)
      .map((e) => ({ from: e.from, to: e.to, chain: Boolean(e.chain || chainKeys.has(`${e.from}:${e.to}`)) }))
      .filter((e, i, edges) => {
        if (!e.chain && exp.arch.dims[e.from] !== exp.arch.dims[e.to]) return false;
        return edges.findIndex((x) => x.chain === e.chain && x.from === e.from && x.to === e.to) === i;
      });
  }
  return exp;
}

function restoreExperiment(exp) {
  normalizeExperiment(exp);
  $("depth").value = Math.max(0, exp.config.depth);
  $("width").value = exp.config.width;
  $("ioDim").value = exp.config.ioDim;
  $("alpha").value = Math.log10(exp.config.alpha);
  $("modelSeed").value = exp.config.modelSeed;
  $("taskSeed").value = exp.config.taskSeed;
  $("optimizer").value = exp.config.optimizer;
  $("lr").value = Math.log10(exp.config.lr);
  $("steps").value = exp.config.steps;
  $("beta1").value = exp.config.beta1;
  $("beta2").value = exp.config.beta2;
  $("epsilon").value = exp.config.epsilon;
  $("weightDecay").value = exp.config.weightDecay;
  state.arch = exp.arch;
  state.dim = state.arch.dims[0];
  state.task = makeTask(exp.config.taskSeed);
  const initialModel = initModel(state.arch, exp.config.modelSeed, exp.config.alpha);
  const restoredModel = cloneModel(initialModel);
  restoredModel.edges.forEach((e, i) => {
    if (e.w && exp.weights?.[i]) e.w.set(exp.weights[i].slice(0, e.w.length));
  });
  state.model = initialModel;
  state.trainedModel = restoredModel;
  const base = lossAndGrad(restoredModel, state.arch, state.task);
  state.theory = computeTheory(initialModel, state.arch, state.task, lossAndGrad(initialModel, state.arch, state.task));
  state.currentRun = { actual: exp.actual, ntk: exp.ntk, color: palette[0], maxStep: exp.config.steps };
  updateLabels();
  updateDiagnostics(base.loss, exp.diagnostics.ntkDeviation);
  drawGraph();
  drawPlot();
}

function renderGallery() {
  const key = $("sortGallery").value;
  const sorted = [...state.gallery].sort((a, b) => {
    normalizeExperiment(a);
    normalizeExperiment(b);
    const va = key === "created" ? a.created : key in a.config ? a.config[key] : a.diagnostics[key];
    const vb = key === "created" ? b.created : key in b.config ? b.config[key] : b.diagnostics[key];
    return key === "created" ? vb - va : (va ?? Infinity) - (vb ?? Infinity);
  });
  $("gallery").innerHTML = "";
  sorted.forEach((exp) => {
    const el = document.createElement("article");
    normalizeExperiment(exp);
    el.className = "card" + (state.overlays.some((o) => o.id === exp.id) ? " selected" : "");
    el.innerHTML = `<div class="card-head"><span>${exp.name}</span><span>${fmt(exp.diagnostics.finalLoss)}</span></div>
      <small>D${exp.config.depth} W${exp.config.width} io ${exp.config.ioDim} alpha ${fmt(exp.config.alpha)} ${exp.config.optimizer} lr ${fmt(exp.config.lr)}</small>
      <canvas class="spark" width="260" height="42"></canvas>`;
    el.addEventListener("click", (ev) => {
      if (ev.shiftKey) {
        const idx = state.overlays.findIndex((o) => o.id === exp.id);
        if (idx >= 0) state.overlays.splice(idx, 1); else state.overlays.push(exp);
        renderGallery(); drawPlot();
      } else restoreExperiment(exp);
    });
    $("gallery").appendChild(el);
    drawSpark(el.querySelector("canvas"), exp.actual);
  });
}

function drawSpark(canvas, pts) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const maxX = Math.max(...pts.map((p) => p.step), 1);
  const maxY = Math.max(...pts.map((p) => p.loss), 1e-12);
  const minY = Math.min(...pts.map((p) => p.loss), maxY * .9);
  ctx.strokeStyle = "#48d6a3"; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((p, i) => {
    const x = (p.step / maxX) * w;
    const y = h - ((p.loss - minY) / Math.max(1e-12, maxY - minY)) * (h - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function updateLabels() {
  const c = cfg();
  $("depthValue").textContent = c.depth;
  $("alphaValue").textContent = c.alpha.toPrecision(3);
  $("lrValue").textContent = c.lr.toPrecision(3);
}

function wire() {
  ["depth", "width", "ioDim", "alpha", "modelSeed", "taskSeed"].forEach((id) => $(id).addEventListener("change", () => rebuild(["depth", "width", "ioDim"].includes(id))));
  ["depth", "alpha", "lr"].forEach((id) => $(id).addEventListener("input", updateLabels));
  $("optimizer").addEventListener("change", () => {
    if ($("optimizer").value === "SGD" && Number($("beta1").value) === 0.9) $("beta1").value = 0;
    if ($("optimizer").value !== "SGD" && Number($("beta1").value) === 0) $("beta1").value = 0.9;
  });
  $("computeBtn").addEventListener("click", recompute);
  $("trainBtn").addEventListener("click", train);
  $("resetBtn").addEventListener("click", () => rebuild(true));
  $("logAxis").addEventListener("change", drawPlot);
  $("logXAxis").addEventListener("change", drawPlot);
  $("clearCompareBtn").addEventListener("click", () => { state.overlays = []; renderGallery(); drawPlot(); });
  $("sortGallery").addEventListener("change", renderGallery);
  $("clearGalleryBtn").addEventListener("click", () => {
    state.gallery = [];
    state.overlays = [];
    saveGallery();
    renderGallery();
    drawPlot();
    setStatus("Gallery cleared");
  });
  $("plot").addEventListener("mousemove", (ev) => {
    if (!state.currentRun?.actual?.length) return;
    const rect = $("plot").getBoundingClientRect();
    const pad = { l: 86, r: 26, t: 30, b: 66 };
    const canvasX = (ev.clientX - rect.left) * ($("plot").width / rect.width);
    const plotWidth = $("plot").width - pad.l - pad.r;
    const maxStep = Math.max(state.currentRun.maxStep || cfg().steps, 1);
    const ratio = Math.max(0, Math.min(1, (canvasX - pad.l) / plotWidth));
    const step = $("logXAxis").checked ? Math.expm1(ratio * Math.log1p(maxStep)) : ratio * maxStep;
    const p = state.currentRun.actual.reduce((best, point) => Math.abs(point.step - step) < Math.abs(best.step - step) ? point : best, state.currentRun.actual[0]);
    $("hoverReadout").textContent = `step ${p.step}  loss ${fmt(p.loss)}`;
  });
}

wire();
loadGallery();
rebuild(true);
renderGallery();
