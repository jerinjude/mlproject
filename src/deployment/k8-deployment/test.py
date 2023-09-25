import requests

input={
    "title":"Diffusion to Vector",
    "description":"Reference implementation of Diffusion2Vec (Complenet 2018) built on Gensim and NetworkX."
}

input={
    "title":"understanding natural language",
    "description":"Abstractive summarization of radiology reports using auto regressive transformer models"
}
url='http://localhost:9696/featurize'
response=requests.post(url,json=input)
print(response.json())

'''from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

model.save('sentence_bert')'''