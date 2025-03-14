from transformers import pipeline


transcriber = pipeline(model="openai/whisper-large-v2")
result = transcriber([
    "./dataset/flac_file/mlk.flac",
    "./dataset/flac_file/1.flac",
    "./dataset/flac_file/2.flac",
    "./dataset/flac_file/3.flac",
    "./dataset/flac_file/4.flac",
])
print(result)

transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
result = transcriber("./dataset/flac_file/mlk.flac")
print(result)