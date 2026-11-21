# Tools

Checks that run against real data, outside `make test`. Neither the API nor the
release build imports anything here (EXP-001). Their findings are written up in
[`docs/experiments/`](../docs/experiments/).

| Tool | Does |
| --- | --- |
| [`backend-math-validation/`](backend-math-validation/) | Compares a running API's time series with an independent computation from the source GeoTIFF |
| [`paleocar-v3-encoding/`](paleocar-v3-encoding/) | Builds and benchmarks COG layout and data type variants (EXP-002 to EXP-004) |
