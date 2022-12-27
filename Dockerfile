FROM python:3.9.15-slim

WORKDIR /classification

COPY src /classification/src
COPY ./setup.py /classification/setup.py

RUN touch README.md && pip install . --no-cache-dir

EXPOSE 5555

CMD [ "sh", "-c", "app --host 0.0.0.0 --port 5555 ${TEST_MODE}" ]