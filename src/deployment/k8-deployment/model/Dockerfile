FROM python:3.11.4-slim

RUN pip install -U pip

WORKDIR /model

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY predict_pipeline.py ./
COPY model1 ./
COPY model_arch.py ./
EXPOSE 9695
CMD ["gunicorn", "--bind=0.0.0.0:9695", "--log-level=debug", "predict_pipeline:app", "--timeout=1120"]