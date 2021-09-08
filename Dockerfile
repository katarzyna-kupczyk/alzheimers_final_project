FROM python:3.8.6-buster

COPY .streamlit/config.toml /.streamlit/config.toml
COPY alzheimers_final_project /alzheimers_final_project
COPY pages /pages
COPY alz_model_h5.h5 /alz_model_h5.h5
COPY app.py /app.py
COPY multipage.py /multipage.py
COPY predict.py /predict.py
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run app.py  --server.port 8501
