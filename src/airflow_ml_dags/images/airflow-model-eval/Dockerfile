FROM airflow-ml-base:light

RUN pip install mlflow
COPY model_eval.py /model_eval.py

ENTRYPOINT ["python", "model_eval.py"]
