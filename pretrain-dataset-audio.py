from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        max_length=100000,
        truncation=True,
    )
    return inputs


data_set = load_dataset("dataset/audio/minds14", name="en-US", split="train")
print(data_set[0]["audio"])
data_set_res = data_set.cast_column("audio", Audio(sampling_rate=16_000))
print(data_set_res[0]["audio"])

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
audio_input = [data_set_res[0]["audio"]["array"]]
feature_res = feature_extractor(audio_input, sampling_rate=16000)
print(feature_res)
print(data_set_res[0]["audio"]["array"].shape)
print(data_set_res[1]["audio"]["array"].shape)
processed_dataset = preprocess_function(data_set_res[:5])
print(processed_dataset)