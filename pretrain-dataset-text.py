from transformers import AutoTokenizer
from datasets import load_dataset, Audio


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)

decoded_output = tokenizer.decode(encoded_input["input_ids"])
print(decoded_output)

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True)
print(encoded_input)

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
print(encoded_input)

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)


dataset = load_dataset("./dataset/PolyAI/minds14", name="en-US", split="train")
print(dataset[0]["audio"])

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
print(dataset[0]["audio"])
