from transformers import (AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor, AutoProcessor, AutoModelForSequenceClassification)


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
feature_extractor = AutoFeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
