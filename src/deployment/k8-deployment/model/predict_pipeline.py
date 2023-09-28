import sys
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import json
from flask import Flask, request, jsonify
import requests

def main(embedding):
    model=torch.load('model1').double()
    model.eval()
    # Perform inference
    with torch.no_grad():
        predictions = model(embedding)

    # `predictions` contains the model's output, which is the probability distribution over classes
    # You can convert this distribution to class labels if needed
    predicted_classes = torch.argmax(predictions, dim=-1)
    return predicted_classes

app=Flask('prediction')

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    input=request.get_json()
    embedding=np.array(input["embedding"])
    pytorch_tensor = torch.from_numpy(embedding).double()
    print(pytorch_tensor)
    prediction_index=main(pytorch_tensor)

    class_names = [
        "computer-vision",
        "graph-learning",
        "reinforcement-learning",
        "natural-language-processing",
        "mlops",
        "time-series",
    ]
    prediction=class_names[prediction_index]
    print(prediction)
    result={'topic':prediction}
    print(result)
    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0',port=9695)

