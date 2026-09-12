FROM python:3.13

ARG ENVIRONMENT
ENV ENVIRONMENT="${ENVIRONMENT}"

RUN apt-get update && apt-get install -y libgdal-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /code

RUN pip install --no-cache-dir --upgrade pip

COPY deploy/requirements /code/requirements

RUN pip install --no-cache-dir \
    -r requirements/base.txt -r requirements/${ENVIRONMENT}.txt

COPY deploy/metadata/${ENVIRONMENT}.yml /code/metadata.yml
COPY deploy/logging/${ENVIRONMENT}.yml /code/config/logging.yml
COPY deploy/settings/${ENVIRONMENT}.yml /code/config/app_settings.yml
COPY deploy/colormaps/custom.json /code/config/colormaps.json

COPY timeseries/app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--reload"]
