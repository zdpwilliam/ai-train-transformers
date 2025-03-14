from huggingface_hub import login, InferenceClient
from transformers import CodeAgent, HfApiEngine, ReactCodeAgent


login("")
client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

def llm_engine_chat(messages, stop_sequences=["Task"]) -> str:
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message.content
    return answer


llm_engine = HfApiEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
agent.run(
    "Could you translate this sentence from French, say it out loud and return the audio.",
    sentence="Où est la boulangerie la plus proche?",
)

agent = CodeAgent(tools=[], add_base_tools=True)
agent.run(
    "Could you translate this sentence from French, say it out loud and return the audio.",
    sentence="Où est la boulangerie la plus proche?",
)

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
agent.run("Why does Mike not know many people in New York?", audio="dataset/mp3/recording.mp3")
print(agent.system_prompt_template)

agent = ReactCodeAgent(tools=[], additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
print(agent.system_prompt_template)
