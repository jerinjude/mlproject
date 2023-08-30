import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

current_script_directory = Path(__file__).resolve().parent
#print(current_script_directory)
parent_directory = Path(current_script_directory.parent / "components/")
#print(parent_directory)
sys.path.append(parent_directory)
sys.path.append("/home/vboxuser/mlprojects/sample/src/components")
sys.path.append("/home/vboxuser/mlprojects/sample/src")



from data_transformation import *

def main(title,desc,bert_model_name,model_path):
    bert_model = SentenceTransformer(bert_model_name)
    text_embd = text_processing_pred(title,desc,bert_model)
    #text_embd=torch.tensor(text_embd)
    #print(text_embd.shape())
    model=torch.load(model_path)
    model.eval()

    # Perform inference
    with torch.no_grad():
        predictions = model(text_embd)
        print(predictions)

    # `predictions` contains the model's output, which is the probability distribution over classes
    # You can convert this distribution to class labels if needed
    predicted_classes = torch.argmax(predictions, dim=-1)

    print('predicted_class',predicted_classes)


main('Diffusion to Vector','Reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX.',"sentence-transformers/all-mpnet-base-v2","/home/vboxuser/mlprojects/sample/src/test/run_dir/model/model1")

