# pip install accelerate
# pip install accelerate bitsandbytes
from transformers import pipeline


pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": False})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
print(output)