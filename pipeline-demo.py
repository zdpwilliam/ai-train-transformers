from transformers import pipeline


classifier = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
result = classifier('We are very happy to introduce pipeline to the transformers repository.')
print(result)