# thai-landslide

Thailand landslide probability, computed entirely in the browser with the NASA LHASA 2.1.1 XGBoost model, over an OpenFreeMap basemap. No backend, no live data feed.

Live site: https://gain9999.github.io/thai-landslide/

## What it does

- Basemap: OpenFreeMap vector positron via MapLibre GL JS 4.7.1 (keyless, https://tiles.openfreemap.org/styles/positron).
- Loads the Thailand static layers at their original 30-arcsecond (~1 km) resolution: 1080 × 1800 cells over 97–106°E, 5.5–20.5°N.
- Runs all 300 XGBoost trees in a Web Worker for every land cell, every time a slider moves, and draws the result as a MapLibre CanvasSource raster layer inside the map, below the basemap's labels — so street names render on top, and the layer pans/zooms/clips with the basemap instead of lagging it as a DOM overlay did.
- Alignment: the data grid is equirectangular, the basemap is Web Mercator, so the overlay is warped row-by-row to Mercator latitudes before drawing. Without this it drifts ~0.8% of the box height (5 px at country zoom, ~80 px at zoom 10, ~315 px at zoom 12). Grid lines and the cell highlight are drawn at the true geographic cell edges (uniform rows, full-height highlight); the colour blocks, being a single warped raster chosen by row centre, can sit up to half a cell off those grid lines. A 3600-row oversampled version that drew the grid at the displayed block edges was tried and reverted: it aligned grid and colour, but made 28 data rows half height and 28 rows 1.5x height, roughly every 50 rows (e.g. row 1058 at lat 11.68).
- Colours are a gradient sampled exactly from NASA's Global Landslide Susceptibility legend: Very Low #0f5fc2 at p=0.10, Low #6a9cd9 at 0.30, Medium #c2c262 at 0.50, High #c2860e at 0.70, Very High #c21010 at 0.90, extended to a darker red #7a0a0a at 1.00 so cells above 0.90 remain distinguishable. Note: those legend colours belong to the susceptibility map, not to v2 probability — the probability anchors above are this site's choice, because v2 does not publish a 5-level probability scale. Cells below the display threshold, sea cells, and (by default) slopes ≤ 10° are left transparent, showing the basemap underneath. Probability is computed for every land cell in the cropped box, including neighbouring countries — the Thailand outline is drawn for reference only and does not clip the model. The stats bar counts only the cells currently on screen (recounted on pan/zoom), not the whole box.
- Display controls: probability opacity slider (default 60%, so street names show through) and 1 km grid lines, shown at zoom 9 and above. Grid lines are vector lines at the true cell edges (not a raster canvas) so they stay sharp and uniform at any zoom, and the probability layer uses nearest-neighbour resampling so cell edges stay crisp.
- Go to cell: type lat/lon in the box above the map (or edit the lat/lon in the cell panel) — the map zooms to at least zoom 13, outlines that 1 km cell in red with a yellow fill, and shows its values. Clicking any cell outlines it the same way.
- Exposure, tapped cell only: tapping a cell shows its population and road length from NASA's exposure files (population.nc4, road_length.nc4, cropped to Thailand, encoded as layers/pop.png and layers/road.png), plus how much of each counts toward exposure at NASA's nested thresholds (low p > 0.1, medium p > 0.5, high p > 0.9). No province or country totals are computed or shown — the inspect panel is the only place exposure appears.
- Mobile: `cooperativeGestures` is on, so one finger scrolls the page and two fingers move the map, plus a “↓ Sliders below” button hovering over the map that jumps to the sliders.

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
| `layers/thai.png` | 8-bit, 255 = inside the GADM Thailand outline (627,555 cells) | exact, retained for the outline; no longer used as a model filter |

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
