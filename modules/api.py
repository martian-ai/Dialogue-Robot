from transformers import pipeline
classifier = pipeline("text-classification",  model="myml/toutiao")
print(classifier("只要关羽不捣乱，峡谷4V5也不怕？"))