[![CI](https://github.com/hyperrealist/visr-tiled/actions/workflows/ci.yml/badge.svg)](https://github.com/hyperrealist/visr-tiled/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/hyperrealist/visr-tiled/branch/main/graph/badge.svg)](https://codecov.io/gh/hyperrealist/visr-tiled)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# visr-tiled

A ViSR-specific packaging of [Tiled](https://blueskyproject.io/tiled/) that adds a
server-side **binning endpoint** for on-the-fly 2-D histogram visualisation of scan
data.

The package ships a patched Tiled 0.2.3 container (changes that have since been merged
upstream) making it a drop-in replacement for the Tiled instance currently deployed on
ViSR.

What       | Where
:---:      | :---:
Source     | <https://github.com/hyperrealist/visr-tiled>
Docker     | `docker run ghcr.io/hyperrealist/visr-tiled:latest`
Releases   | <https://github.com/hyperrealist/visr-tiled/releases>

---

## Binning endpoint

### Overview

The `/api/v1/binned/{uid}` endpoint bins a ViSR scan into a 2-D image on the server
and returns per-channel weighted histograms.  Positions can be derived either from the
**recorded readback values** (default) or from the **ScanSpec setpoints** stored in the
run's start document.

The endpoint is live at:

```
http://172.23.71.100:8000/api/v1/binned/{uid}
```

where `{uid}` is the unique node ID from Tiled.

### Response schema

```json
{
  "RedTotal":   [[123, 456, ...], ...],
  "GreenTotal": [[789, 123, ...], ...],
  "BlueTotal":  [[456, 789, ...], ...],
  "x_limits":   [0.0, 0.5, ...],
  "y_limits":   [0.0, 0.5, ...]
}
```

- `RedTotal`, `GreenTotal`, `BlueTotal` — 2-D arrays (lists of lists) containing the
  mean channel value in each histogram bin.
- `x_limits`, `y_limits` — 1-D arrays of bin *edges* (length = number of bins + 1)
  along each axis, as returned by `numpy.histogram2d`.

### Query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_dim_index` | integer | `0` | Index into the position array to use as the x axis. |
| `y_dim_index` | integer | `1` | Index into the position array to use as the y axis. |
| `xmin` | float | — | Lower bound of the x histogram range. When combined with `xmax`, `ymin`, and `ymax`, passed as the `range` argument to `numpy.histogram2d`. |
| `xmax` | float | — | Upper bound of the x histogram range. |
| `ymin` | float | — | Lower bound of the y histogram range. |
| `ymax` | float | — | Upper bound of the y histogram range. |
| `width` | integer | — | Number of bins along the x axis. Must be set together with `height`. |
| `height` | integer | — | Number of bins along the y axis. Must be set together with `width`. |
| `setpoints` | boolean | `false` | If `true`, derive positions from the ScanSpec setpoints stored in the run's start document instead of the recorded readback values. |
| `slice_dim` *(experimental)* | string (repeatable) | — | Filter data points to a slice along a dimension that is neither x nor y. Format: `dim:center:thickness`, where `dim` is the integer dimension index, `center` is the centre of the slice, and `thickness` is the half-width — only points satisfying `\|position − center\| ≤ thickness` are included. The parameter may be repeated to apply multiple slices simultaneously. |

### Example request

```
GET http://172.23.71.100:8000/api/v1/binned/aaf5e459-eb01-487c-8f90-a9468d4a2852
    ?width=256&height=256
    &xmin=-1.0&xmax=1.0&ymin=-1.0&ymax=1.0
    &setpoints=false
```
