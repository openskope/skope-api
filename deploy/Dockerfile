FROM python:3.8

COPY requirements.txt /code/
COPY

WORKDIR /code

RUN pip install -r requirements.txt

COPY app /code/

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "app.main:app"]
