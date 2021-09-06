FROM python:3.8.6-buster

COPY alzheimers_final_project /alzheimers_final_project
COPY requirements.txt /requirements.txt
COPY predict.py /predict.py
COPY api /api
COPY model.joblib /model.joblib


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
