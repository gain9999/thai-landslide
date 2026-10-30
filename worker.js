// Worker for thai-landslide: evaluates the NASA LHASA 2.1.1 XGBoost model
// (300 depth-2 trees, binary logistic) over the Thailand 1080x1800 grid.
// Model conversion verified against the real XGBoost Booster in Python:
// max absolute probability difference < 4e-7 over 200 random land cells,
// across three scenarios. Base score is used as logit(base_score).
'use strict';

let W = 1080, H = 1800;
let slopeArr, lithArr, pgaArr, maskArr, p99Arr, thaiArr;
let treeOffsets, splitIdx, splitCond, leftArr, rightArr, baseLogit = 0;
let lastProb = null, lastParams = null;

// Gradient colour stops for probability p: [p, r, g, b]. Colours are
// sampled exactly from NASA's Global Landslide Susceptibility legend
// (see README): Very Low #0f5fc2, Low #6a9cd9, Medium #c2c262,
// High #c2860e, Very High #c21010. Those legend colours belong to the
// susceptibility map, not to v2 probability — the probability ranges
// assigned to them here are ours, chosen so the anchors land at round
// probabilities (0.10 / 0.30 / 0.50 / 0.70 / 0.90). The gradient is
// extended to a darker red at 1.00 so cells above 0.90 are still
// distinguishable from each other instead of all rendering identical.
const STOPS = [
  [0.10, 15, 95, 194],
  [0.30, 106, 156, 217],
  [0.50, 194, 194, 98],
  [0.70, 194, 134, 14],
  [0.90, 194, 16, 16],
  [1.00, 122, 10, 10]
];

function colour(p) {
  if (p <= STOPS[0][0]) return STOPS[0].slice(1);
  for (let i = 1; i < STOPS.length; i++) {
    if (p <= STOPS[i][0]) {
      const a = STOPS[i - 1], b = STOPS[i];
      const t = (p - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
        Math.round(a[3] + (b[3] - a[3]) * t)
      ];
    }
  }
  return STOPS[STOPS.length - 1].slice(1);
}

self.onmessage = function (e) {
  const d = e.data;
  if (d.type === 'init') {
    W = d.w; H = d.h;
    slopeArr = d.slope; lithArr = d.lith; pgaArr = d.pga;
    maskArr = d.mask; p99Arr = d.p99; thaiArr = d.thai;
    // Flatten trees into typed arrays for speed.
    const trees = d.model.trees;
    const offsets = [0];
    let total = 0;
    for (const t of trees) { total += t.s.length; offsets.push(total); }
    treeOffsets = new Int32Array(offsets);
    splitIdx = new Uint8Array(total);
    splitCond = new Float32Array(total);
    leftArr = new Int32Array(total);
    rightArr = new Int32Array(total);
    let k = 0;
    for (const t of trees) {
      for (let i = 0; i < t.s.length; i++) {
        splitIdx[k] = t.s[i]; splitCond[k] = t.c[i];
        leftArr[k] = t.l[i]; rightArr[k] = t.r[i]; k++;
      }
    }
    baseLogit = Math.log(d.model.base_score / (1 - d.model.base_score));
    self.postMessage({ type: 'ready' });
    return;
  }
  if (d.type === 'viewport') {
    // Stats for the cells currently on screen only. The main thread
    // sends the map's lng/lat bounds; the worker counts from the last
    // computed probability grid, applying the same slope filter and
    // display threshold as the full-map stats. No model re-run needed.
    if (!lastProb || !lastParams) return;
    const b = d.bounds;
    const x0 = Math.max(0, Math.floor((b.west - 97) / 9 * W));
    const x1 = Math.min(W - 1, Math.ceil((b.east - 97) / 9 * W) - 1);
    const y0 = Math.max(0, Math.floor((20.5 - b.north) / 15 * H));
    const y1 = Math.min(H - 1, Math.ceil((20.5 - b.south) / 15 * H) - 1);
    let shown = 0, above50 = 0, above90 = 0, sum = 0, maxP = 0;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!maskArr[i]) continue;
        const slope = lastParams.override ? lastParams.slopeOv : slopeArr[i];
        if (lastParams.slopeFilter && slope <= 10) continue;
        const prob = lastProb[i];
        if (prob < lastParams.minP) continue;
        shown++; sum += prob; if (prob > maxP) maxP = prob;
        if (prob >= 0.5) above50++;
        if (prob >= 0.9) above90++;
      }
    }
    self.postMessage({
      type: 'viewportStats',
      stats: { shown, above50, above90, mean: shown ? sum / shown : 0, max: maxP }
    });
    return;
  }
  if (d.type === 'compute') {
    const p = d.params;
    lastParams = p;
    lastProb = new Float32Array(W * H);
    const out = new Uint8ClampedArray(W * H * 4);
    let shown = 0, above50 = 0, above90 = 0, sum = 0, maxP = 0;
    const feat = new Float32Array(6);
    const nTrees = treeOffsets.length - 1;
    for (let i = 0; i < W * H; i++) {
      // All land cells in the cropped box, not just Thailand — the
      // Thailand mask (thaiArr) is no longer a filter here.
      if (!maskArr[i]) continue;
      const slope = p.override ? p.slopeOv : slopeArr[i];
      if (p.slopeFilter && slope <= 10) continue;
      const lith = p.override ? p.lithOv : lithArr[i];
      const pga = p.override ? p.pgaOv : pgaArr[i];
      feat[0] = lith; feat[1] = slope; feat[2] = pga;
      feat[3] = p.gwet; feat[4] = p.ante;
      feat[5] = p.rainMm / p99Arr[i];
      let totalW = 0;
      for (let t = 0; t < nTrees; t++) {
        let node = treeOffsets[t];
        const end = treeOffsets[t + 1];
        // Walk: node index here is absolute; leaf is where left == -1.
        // Tree nodes are stored contiguously, but child indices are
        // relative to the tree, so add the tree offset.
        let rel = 0;
        while (true) {
          const abs = node + rel;
          const lc = leftArr[abs];
          if (lc === -1) { totalW += splitCond[abs]; break; }
          rel = (feat[splitIdx[abs]] < splitCond[abs]) ? lc : rightArr[abs];
        }
        void end;
      }
      const prob = 1 / (1 + Math.exp(-(baseLogit + totalW)));
      lastProb[i] = prob;
      if (prob < p.minP) continue;
      const c = colour(prob);
      const o = i * 4;
      out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 255;
      shown++; sum += prob; if (prob > maxP) maxP = prob;
      if (prob >= 0.5) above50++;
      if (prob >= 0.9) above90++;
    }
    self.postMessage({
      type: 'result', image: out.buffer,
      stats: { shown, above50, above90, mean: shown ? sum / shown : 0, max: maxP }
    }, [out.buffer]);
  }
};
