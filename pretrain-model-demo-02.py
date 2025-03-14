import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


data_set = load_dataset("dataset/yelp_review_full")
data_set_train = data_set["train"][100]
print(data_set_train)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_datasets = data_set.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
print(tokenized_datasets)
print(small_eval_dataset)
print(small_train_dataset)

training_args = TrainingArguments(output_dir="dataset/trainer")
print(training_args)

metric = evaluate.load("accuracy")
print(metric)

training_args = TrainingArguments(output_dir="dataset/trainer", eval_strategy="epoch")
print(training_args)

model = AutoModel.from_pretrained(pretrained_model_name_or_path="distilbert/distilbert-base-uncased", trust_remote_code=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=metric,
)
print(trainer)