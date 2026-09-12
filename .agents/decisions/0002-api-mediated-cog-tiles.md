# 0002: Mediate COG Tiles Through Skope API

- Status: Accepted
- Date: 2026-08-11

## Context

Direct client access to COG storage would expose storage layout, shift decoding work to browsers, and make rendering policy and observability client-specific.

## Decision

Expose XYZ raster tiles through `skope-api`. The API validates dataset and variable identifiers, resolves the requested timestep through the dataset lookup, and streams the result from an internal TiTiler service. TiTiler is not exposed on a host port.

## Consequences

- The public tile contract includes dataset, variable, year, XYZ coordinates, colormap, and rescale parameters.
- Storage and TiTiler remain implementation details behind the API.
- API and TiTiler health are both required for raster visualization.

