from transformers import pipeline


def data():
    for i in range(5):
        yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2")
generated_characters = ""
for out in pipe(data()):
    generated_characters += out[0]["generated_text"]
print(generated_characters)