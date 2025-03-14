from transformers import pipeline


vqa = pipeline(model="impira/layoutlm-document-qa")
output = vqa(image="./dataset/img/invoice.png", question="Can you describe this picture in detail?")
print(output)
output[0]["score"] = round(output[0]["score"], 3)