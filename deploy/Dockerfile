FROM python:3.8

ARG ENVIR
ENV SETTINGS_MODULE="app.settings.${ENVIR}"

COPY deploy/requirements /code/deploy/requirements
COPY deploy/logging/${ENVIR}.yml /code/deploy/logging/${ENVIR}.yml

WORKDIR /code

RUN pip install -r deploy/requirements/base.txt -r deploy/requirements/${ENVIR}.txt

COPY app /code/

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.main:app"]
