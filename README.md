# thai-landslide

Thailand landslide probability, computed entirely in the browser with the NASA LHASA 2.1.1 XGBoost model, over an OpenFreeMap basemap. No backend, no live data feed.

Live site: https://gain9999.github.io/thai-landslide/

## What it does

- Basemap: OpenFreeMap vector positron via MapLibre GL JS 4.7.1 (keyless, https://tiles.openfreemap.org/styles/positron).
- Loads the Thailand static layers at their original 30-arcsecond (~1 km) resolution: 1080 × 1800 cells over 97–106°E, 5.5–20.5°N.
- Runs all 300 XGBoost trees in a Web Worker for every land cell, every time a slider moves, and draws the result as a transparent canvas overlay positioned with MapLibre's `project()` on every move/zoom — so the basemap's streets, labels and borders stay visible wherever probability is not coloured.
- Colours cells by probability (0.1 yellow → 0.9+ dark red). Cells below the display threshold, sea cells, outside-Thailand cells, and (by default) slopes ≤ 10° are left transparent, showing the basemap underneath.

## Sliders — all model inputs

Scenario (dynamic) inputs, set by the user:

- Today's rain, mm (0–300). The model uses rain ÷ p99 per cell, where p99 is the local 99th-percentile daily rain.
- Previous 2 days' rain / antecedent, mm (0–400).
- Soil moisture, % (0–100). Model value = % ÷ 100 (SMAP `sm_profile_wetness` / GEOS `gwetprof`).

Terrain (static) inputs, real per-cell values by default, with a what-if override:

- Slope, ° (0–70)
- Lithology code (0–7, codes from the LHASA layer; the file carries no rock-type names)
- Peak ground acceleration, PGA, g (0–0.50)

Turning on "Override terrain" replaces every cell's real slope/lithology/PGA with the slider values, so you can test terrain sensitivity. With it off, those sliders are disabled and the map uses the real layers.

Presets: Dry day, Phuket 26 Oct-like (117 mm, 80 mm antecedent, 75% moisture), Wet monsoon, Extreme.

Tap/click any cell to see its lat/lon, real terrain values, p99, rain ratio and exact probability.

## How the model was converted

- Source: `model.json` from NASA LHASA v2.1.1 (`model.json` in the LHASA repo, 300 trees, binary logistic, base score 0.1246).
- Stripped to splits, thresholds, child indices and leaf weights only: 45 KB (`model.json` in this repo).
- Prediction in the browser: `p = sigmoid(logit(base_score) + sum of leaf weights)`, walking each tree left if `feature < threshold`, else right.
- Verified in Python against the real XGBoost Booster (v3.4.1) on 200 random Thailand land cells across three scenarios (dry / Phuket-like / extreme): max absolute probability difference < 4×10⁻⁷. The logit-base-score form matches; using base score raw does not (mean error 0.084).

## Layer encoding (so the browser stays small)

Total site is about 2.8 MB. Layers are PNGs, read back through canvas:

| File | Encoding | Round-trip error |
|---|---|---|
| `layers/slope.png` | 16-bit, R=high byte, G=low byte, value = int ÷ 100 (°) | ≤ 0.005° |
| `layers/pga.png` | 16-bit, value = int ÷ 10000 (g) | ≤ 0.00005 g |
| `layers/p99.png` | 16-bit, value = int (mm), p99 resampled nearest to the 1 km grid | ≤ 0.5 mm |
| `layers/lith.png` | 8-bit code 0–7 | exact |
| `layers/mask.png` | 8-bit, 255 = land | exact |
| `layers/thai.png` | 8-bit, 255 = inside the GADM Thailand outline (627,555 cells) | exact |

`layers/thai.png` was derived from the GADM v4.1 Thailand level-0 outline, simplified to 0.01°. The page no longer draws the outline itself — country borders come from the OpenFreeMap basemap.

## What this is not

- Not live data. Rain, antecedent rain and soil moisture here are scenario values, not today's IMERG/SMAP. A live version would need a daily feed for those three.
- Not a flood forecast, and not a measured landslide frequency for a specific hillside. Probabilities are the model's outputs; under extreme sliders the model saturates near 1, which is what the model says, not a certainty claim.
- Rain and soil moisture in the real LHASA system are natively ~11 km (IMERG 0.1°) and 9 km (SMAP), resampled to the 1 km grid. So the 1 km detail in this map comes from slope, lithology and PGA; rain/moisture patterns are effectively ~10 km.

## Run locally

```sh
python3 -m http.server 8000
# open http://localhost:8000/
```

A plain `file://` open will fail on the Web Worker; use a local server.

## Sources

- NASA LHASA: https://github.com/nasa/LHASA (v2.1.1)
- LHASA static data package (static.zip, 2024-08-07): Slope, Lithology, PGA, mask, p99
- Basemap: OpenFreeMap, https://openfreemap.org (© OpenMapTiles, © OpenStreetMap contributors)
- Outline: GADM v4.1, https://gadm.org/

Independent client-side build. Not an official NASA product.
