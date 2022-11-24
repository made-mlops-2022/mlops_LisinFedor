FROM python:3.9.15-slim

WORKDIR /classification

COPY src /classification/src
COPY ./setup.py /classification/setup.py

RUN touch README.md && pip install . --no-cache-dir
CMD [ "sh", "-c", "app --host 0.0.0.0 --port 80 ${TEST_MODE}" ]