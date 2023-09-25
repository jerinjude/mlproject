FROM python:3.11.4-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY predict_pipeline.py ./
COPY model1 ./
COPY model_arch.py ./
COPY setup.py ./
EXPOSE 9696
RUN python setup.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict_pipeline:app"]





