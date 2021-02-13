FROM python:3.8

COPY requirements /code/requirements

WORKDIR /code

RUN pip install -r requirements/base.txt -r requirements/dev.txt

COPY app /code/

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.main:app"]
