from datasets import load_dataset, Audio
from transformers import AutoProcessor


def prepare_dataset(example):
    audio = example["audio"]
    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))
    return example


lj_speech = load_dataset("dataset/video/lj_speech", split="train", trust_remote_code=True)
lj_speech = lj_speech.map(remove_columns=["file", "id", "normalized_text"])
print(lj_speech[0]["audio"])
print(lj_speech[0]["text"])

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
lj_speech = lj_speech.cast_column("audio", Audio(sampling_rate=16_000))
lj_speech_res = prepare_dataset(lj_speech[0])
print(lj_speech_res)