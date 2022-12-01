FROM airflow-ml-base:light

RUN pip install sdv
COPY new_data_imitation.py /new_data_imitation.py

ENTRYPOINT ["python", "new_data_imitation.py"]
