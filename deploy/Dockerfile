FROM python:3.13

ARG ENVIRONMENT
ENV ENVIRONMENT="${ENVIRONMENT}"

RUN apt-get update && apt-get install -y libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install --no-cache-dir --upgrade pip

COPY deploy/requirements /code/requirements

RUN pip install --no-cache-dir \
    -r requirements/base.txt -r requirements/${ENVIRONMENT}.txt

COPY deploy/logging/${ENVIRONMENT}.yml /code/config/logging.yml
COPY deploy/settings/${ENVIRONMENT}.yml /code/config/app_settings.yml
COPY deploy/releases/${ENVIRONMENT}.yml /code/config/releases.yml
COPY deploy/display /code/config/display

COPY timeseries/app /code/app

# Run unprivileged, as the same uid as TiTiler, so both read releases the same
# way. The app's own files stay root-owned and read-only to it.
RUN useradd --uid 1000 --create-home skope
USER skope

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
