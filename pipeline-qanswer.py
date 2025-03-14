from transformers import pipeline


# 使用问答流水线
question_answerer = pipeline('question-answering')
result = question_answerer({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline has been included in the huggingface/transformers repository'
})
print(result)