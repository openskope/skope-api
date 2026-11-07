# Backend math validation

- Run: 2026-11-06, against the dev API and the `paleocar_v3-r-2026.10.07` release (`ppt_annual`,
  `gdd_cotton_annual`)
- Harness: [`experiments/backend-math-validation/`](../../experiments/backend-math-validation/)
- Result: **three bugs found and fixed; after the fixes, 16 of 16 extractions and 1,440 of 1,440 analyze cases match
  the reference**

## Question

Does the API compute what it claims? An extraction turns a study area and a time range into one value per timestep
(the zonal mean or median), and `/analyze` applies z-scores and smoothers to it. Errors here are silent: a plot looks
plausible whether or not the right cells, the right years or the right windows went into it.

## Method

`validate.py` computes every value independently and compares it with the API's:

- **Source:** the variable's source GeoTIFF, one file with all 1,898 timesteps and each year in its band
  description, rather than the release's COGs. A misordered band, a wrong chunk boundary or a shifted year in the
  release or in the API's time-to-band rule shows up as a mismatch.
- **Cells:** a point-in-polygon test on every cell center (shapely), and the touched cells when no center is inside.
- **Statistics:** NumPy in float64 per timestep, skipping nodata.
- **Transforms and smoothers:** plain loops over the definitions in the script's docstring, applied to the whole
  axis and then cut to the range.

The cases: 8 areas (see the [harness README](../../experiments/backend-math-validation/README.md)) × 2 variables, each
extracted over the whole axis and compared, then 90 `/analyze` cases each: mean and median × 5 transforms (none,
z-score on the selected range, fixed reference 1000–1100, fixed reference inside the empty years 0450–0500, moving
window 10) × 3 smoothers (none, trailing 5, centered 7) × 3 ranges (all, 0350–0700 across the empty years 0417–0589
and several files, 1500–1600). Each case compares values, null positions, timesteps, the series' time range, summary
statistics and the cell count.

`tests/core/test_math_reference.py` holds the same checks on synthetic rasters for the test suite; each test that
covers a fixed bug fails on the code before the fix.

## Findings

### Bugs, fixed

1. **The read window dropped the last row or column of cells.** `resolve_spatial_window` rounded the window's length
   and then its offset. When an area's bounds started more than half a cell into a cell, the window ended one cell
   short, and cells whose centers lie inside the area were never read. Of the 8 areas, 3 lost cells: a 20 × 20 box
   gave 361 cells instead of 400, a 100 × 100 box 9,801 instead of 10,000, and the 5 × 5 test box 16 instead of 25.
   Values differed by up to 1.5%. Any user-drawn area could be affected, always on its east or south edge. The window
   now runs from the floor of the bounds' start to the ceiling of their end. `estimate_cell_count` uses the same
   window, so the size check before an extraction counts one more row and column than before.
2. **Zonal means were summed in float32.** Their relative error was about 6e-8, small by itself, but moving z-scores
   divide by a window's spread and turned it into differences of up to 4e-4. Means now accumulate in float64; the
   data still reads as float32, so memory is unchanged. Medians were already exact.
3. **A fixed-interval z-score could invent values.** With a reference period whose values were all equal it returned
   0 for every timestep, filling in the empty years. With a reference period that had no values (inside 0417–0589) it
   also returned 0 everywhere. A constant reference now gives null everywhere: the z-score divides by the reference's
   spread, which is 0, so it has no value, as a moving z-score already gave for a window without spread. A reference
   without values is refused (see below).

### Confirmed

- Every timestep's value matches the source for the right year, across all 19 files of each variable: the release's
  layout and the API's time-to-band rule agree with the source's band descriptions.
- Cell membership matches the cell-center rule for boxes, a triangle, two disjoint polygons and a box on the grid's
  edge, and the touched-cell fallback for a point and a sliver.
- Moving and fixed z-scores, trailing and centered averages, the order "transform and smooth the whole axis, then
  cut", the null positions around the empty years, and the summary statistics all match their definitions.

### Decided after the run: moving windows need every value

The run found smoothing windows near the empty years and at the axis ends averaging fewer values than the window
width: a 7-year centered average at 0414 averaged 0411–0416, six values. Smoothing now follows the moving z-score's
rule, as the legacy API did: a window gives a value only where every timestep it covers has one, so each gap widens
by the window. A 7-year centered average is null for 0414–0416 and
0590–0592, and a moving z-score of width 10 for 0590–0599; the rest of the axis keeps its values. The harness and the
tests were changed to this definition, and a rerun matched 16 of 16 extractions and 1,440 of 1,440 analyze cases.

### Decided after the run: population standard deviation throughout

The z-scores used the population standard deviation (n) and the summary statistics the sample one (n − 1), as
pandas' default left them. The summary statistics now use n as well, as the legacy API did; a single value has a
standard deviation of 0, no longer none. skopeui's statistics on the visualize page still use n − 1. A rerun matched
16 of 16 extractions and 1,440 of 1,440 analyze cases.

### Decided after the run: requests that can't give a value are refused

- A z-score needs at least 2 values: a moving window of at least 2 time steps, and a fixed reference period with at
  least 2 time steps that have values. A reference period inside 0417–0589 is now a 422 that names the period, not
  a series of nulls.
- A window must fit in the dataset's axis: a smoothing window at most its number of time steps, a moving z-score's
  at most one fewer. This replaces the fixed maximum of 200, which dated from when the API padded each read by the
  window.
- `/timeseries/analyze` answers 410 when the stored extraction can't serve the request, so the client extracts again;
  422 always means the request needs changing.

A rerun matched 16 of 16 extractions and 1,440 of 1,440 analyze cases; the 288 with the reference inside the empty
years are now checked to be refused.

### Kept as the legacy API had it: every cell weighs the same

Zonal statistics weight every cell equally, as the legacy API did; the legacy user guide does not mention weighting. The grid is
geographic, so a cell's area shrinks with latitude: about 15% between the dataset's southern and northern edges, and a
tall area's mean leans slightly towards its northern cells.

## Scope

- The run used the dev release (UInt32, 100 timesteps per file), not the staging release (UInt16, 250 per file). The
  extraction code is the same for both, and the encoding experiment found identical results across layouts, but
  running `validate.py` against staging would confirm it on the release users see.
- Scale and offset are not covered: the API refuses to serve a variable whose scale or offset is not identity.
- Static datasets and categorical variables are not served yet and are not covered.
