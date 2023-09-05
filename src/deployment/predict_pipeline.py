import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn as nn
import json
from flask import Flask, request, jsonify
    
def text_processing_pred(title,desc,bert_model):
    text = title + " " + desc
    text = featurization(text, bert_model)
    text = torch.tensor(text)
    return text

def featurization(text, model):
    embeddings = model.encode(text,show_progress_bar=False)
    return embeddings

def main(title,desc):
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    #bert_model=SentenceTransformer('sentence_bert/pytorch_model.bin')
    text_embd = text_processing_pred(title,desc,bert_model)
    #text_embd=torch.tensor(text_embd)
    #print(text_embd.shape())
    model=torch.load('model1')
    model.eval()

    # Perform inference
    with torch.no_grad():
        predictions = model(text_embd)

    # `predictions` contains the model's output, which is the probability distribution over classes
    # You can convert this distribution to class labels if needed
    predicted_classes = torch.argmax(predictions, dim=-1)
    class_names = [
            "computer-vision",
            "graph-learning",
            "reinforcement-learning",
            "natural-language-processing",
            "mlops",
            "time-series",
        ]
    

    return class_names[predicted_classes]

app=Flask('topic-prediction')

@app.route('/predict',methods=['POST'])
def predict_endpoint():

    input=request.get_json()
    title=input["title"]
    description=input["description"]
    prediction=main(title,description)
    result={'topic':prediction}

    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0',port=9696)

