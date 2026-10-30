// Worker for thai-landslide: evaluates the NASA LHASA 2.1.1 XGBoost model
// (300 depth-2 trees, binary logistic) over the Thailand 1080x1800 grid.
// Model conversion verified against the real XGBoost Booster in Python:
// max absolute probability difference < 4e-7 over 200 random land cells,
// across three scenarios. Base score is used as logit(base_score).
'use strict';

let W = 1080, H = 1800;
let slopeArr, lithArr, pgaArr, maskArr, p99Arr, thaiArr;
let treeOffsets, splitIdx, splitCond, leftArr, rightArr, baseLogit = 0;

// Discrete probability bins: [upper edge, r, g, b]. ColorBrewer YlOrRd
// 9-class, with finer bins where the model actually clusters in wet
// scenarios (0.78-0.95 holds most cells; 0.95-1.00 is rare but real in
// extreme scenarios). Discrete steps differentiate far better than a
// smooth gradient, especially under the overlay's transparency.
const BINS = [
  [0.20, 255, 255, 204],
  [0.35, 255, 237, 160],
  [0.50, 254, 217, 118],
  [0.65, 254, 178, 76],
  [0.78, 253, 141, 60],
  [0.86, 252, 78, 42],
  [0.91, 227, 26, 28],
  [0.95, 189, 0, 38],
  [1.01, 128, 0, 38]
];

function colour(p) {
  for (let i = 0; i < BINS.length; i++) {
    if (p < BINS[i][0]) return BINS[i].slice(1);
  }
  return BINS[BINS.length - 1].slice(1);
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
  if (d.type === 'compute') {
    const p = d.params;
    const out = new Uint8ClampedArray(W * H * 4);
    let shown = 0, above50 = 0, above90 = 0, sum = 0, maxP = 0;
    const feat = new Float32Array(6);
    const nTrees = treeOffsets.length - 1;
    for (let i = 0; i < W * H; i++) {
      if (!maskArr[i] || !thaiArr[i]) continue;
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
