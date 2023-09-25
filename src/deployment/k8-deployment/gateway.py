import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn as nn
import json
from flask import Flask, request, jsonify
import requests

def text_processing_pred(title,desc,bert_model):
    text = title + " " + desc
    text = featurization(text, bert_model)
    text = torch.tensor(text)
    return text

def featurization(text, model):
    embeddings = model.encode(text,show_progress_bar=False)
    return embeddings

def feature(title,desc):
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    text_embd = text_processing_pred(title,desc,bert_model)
    return text_embd

app=Flask('gateway')
@app.route('/featurize',methods=['POST'])
def recieve_text_featurize():
    input=request.get_json()
    title=input["title"]
    description=input["description"]
    text_embd=feature(title,description)
    #numpy_array = text_embd.numpy().tolist()
    result={'embedding':text_embd.numpy().tolist()}
    print('embedding',result)
    url='http://localhost:9695/predict'
    response=requests.post(url,json=result)
    print(response)
    if response.status_code == 200:
    # Extract JSON-serializable data from the response
        response_data = response.json()
        print(response_data)
        return jsonify(response_data)
    else:
        return jsonify(error=f"HTTP Error: {response.status_code}")


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)