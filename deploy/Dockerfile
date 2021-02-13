FROM python:3.8

COPY requirements.txt metadata.yml main.py /code/

WORKDIR /code

RUN pip install -r requirements.txt

CMD ["uvicorn", "--reload", "--host", "0.0.0.0", "main:app"]
