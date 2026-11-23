# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

ARG ENVIRONMENT
ARG REQUIREMENTS=production
ENV ENVIRONMENT="${ENVIRONMENT}"

# Rasterio's bundled GDAL still links to the system XML parser.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install exactly what timeseries/app/uv.lock records, with the REQUIREMENTS
# group (dev or production). uv and its temporary files exist only for this step.
# Rasterio wheels bundle GDAL, so no system GDAL or build tools are needed.
# Package installers and their bundled dependencies are not needed at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=from=ghcr.io/astral-sh/uv:0.13.8,source=/uv,target=/bin/uv \
    --mount=type=bind,source=timeseries/app/pyproject.toml,target=/deps/pyproject.toml \
    --mount=type=bind,source=timeseries/app/uv.lock,target=/deps/uv.lock \
    --mount=type=tmpfs,target=/tmp \
    uv export --project /deps --format requirements.txt --locked --no-emit-project \
    --no-hashes --no-default-groups --group "${REQUIREMENTS}" --output-file /tmp/requirements.txt \
    && uv pip install --system --only-binary :all: -r /tmp/requirements.txt \
    && python -m pip uninstall --yes pip \
    && python -c 'import ensurepip, shutil; shutil.rmtree(ensurepip.__path__[0])'

COPY deploy/logging/${ENVIRONMENT}.yml /code/config/logging.yml
COPY deploy/settings/${ENVIRONMENT}.yml /code/config/app_settings.yml
COPY deploy/releases/${ENVIRONMENT}.yml /code/config/releases.yml
COPY deploy/display /code/config/display

ARG SKOPE_BUILD_ID=development
ENV SKOPE_BUILD_ID="${SKOPE_BUILD_ID}"

COPY timeseries/app /code/app

# Run unprivileged, as the same uid as TiTiler, so both read releases the same
# way. The app's own files stay root-owned and read-only to it.
RUN chmod -R a+rX /code/app /code/config \
    && useradd --uid 1000 --create-home skope
USER skope

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
