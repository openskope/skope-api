# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

ARG ENVIRONMENT
ARG REQUIREMENTS=production
ENV ENVIRONMENT="${ENVIRONMENT}"

# Rasterio's bundled GDAL still links to the system XML parser.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY deploy/requirements /code/requirements

# Rasterio wheels bundle GDAL, so no system GDAL or build tools are needed.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: \
    -r requirements/base.txt -r requirements/${REQUIREMENTS}.txt

COPY deploy/logging/${ENVIRONMENT}.yml /code/config/logging.yml
COPY deploy/settings/${ENVIRONMENT}.yml /code/config/app_settings.yml
COPY deploy/releases/${ENVIRONMENT}.yml /code/config/releases.yml
COPY deploy/display /code/config/display

COPY timeseries/app /code/app

# Run unprivileged, as the same uid as TiTiler, so both read releases the same
# way. The app's own files stay root-owned and read-only to it.
RUN useradd --uid 1000 --create-home skope
USER skope

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
