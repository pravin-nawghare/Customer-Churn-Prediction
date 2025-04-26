FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]