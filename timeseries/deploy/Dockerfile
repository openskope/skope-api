FROM python:3.8

ARG ENVIR
ENV ENVIR="${ENVIR}"

COPY deploy/requirements /code/deploy/requirements
COPY deploy/logging /code/deploy/logging

WORKDIR /code

RUN pip install -r deploy/requirements/base.txt -r deploy/requirements/${ENVIR}.txt

COPY app metadata.yml /code/

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.main:app"]
