FROM python:3.11.4-slim

RUN pip install -U pip

WORKDIR /gateway

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY gateway.py ./
COPY setup.py ./
EXPOSE 9696
RUN python setup.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "--log-level=debug", "gateway:app", "--timeout=1120"]