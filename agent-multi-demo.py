from huggingface_hub import login, list_models
from transformers import Tool, load_tool, CodeAgent, stream_to_gradio
import gradio as gr
from gradio_tools import StableDiffusionPromptGeneratorTool
from transformers.agents import ReactCodeAgent, HfApiEngine, DuckDuckGoSearchTool, ManagedAgent


login("")
llm_engine = HfApiEngine()
web_agent = ReactCodeAgent(tools=[DuckDuckGoSearchTool()], llm_engine=llm_engine)
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="web_search",
    description="Runs web searches for you. Give it your query as an argument."
)
manager_agent = ReactCodeAgent(
    tools=[], llm_engine=llm_engine, managed_agents=[managed_web_agent]
)
manager_agent.run("Who is the CEO of Hugging Face?")


class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.
    """
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id


tool = HFModelDownloadsTool()
tool.push_to_hub("deepen-zhang/hf-model-downloads")

model_download_tool = load_tool("m-ric/hf-model-downloads")
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-dev", name="image_generator", description="Generate an image from a prompt")
image_generation_tool("A sunny beach")

agent = ReactCodeAgent(tools=[image_generation_tool])
agent.run(
    "Improve this prompt, then generate an image of it.", prompt='A rabbit wearing a space suit'
)

gradio_prompt_generator_tool = StableDiffusionPromptGeneratorTool()
prompt_generator_tool = Tool.from_gradio(gradio_prompt_generator_tool)


# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image")

llm_engine = HfApiEngine("meta-llama/Meta-Llama-3-70B-Instruct")

# Initialize the agent with the image generation tool
agent = ReactCodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)


def interact_with_agent(task):
    messages = []
    messages.append(gr.ChatMessage(role="user", content=task))
    yield messages
    for msg in stream_to_gradio(agent, task):
        messages.append(msg)
        yield messages + [
            gr.ChatMessage(role="assistant", content="⏳ Task not finished yet!")
        ]
    yield messages


with gr.Blocks() as demo:
    text_input = gr.Textbox(lines=1, label="Chat Message", value="Make me a picture of the Statue of Liberty.")
    submit = gr.Button("Run illustrator agent!")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
    )
    submit.click(interact_with_agent, [text_input], [chatbot])


if __name__ == "__main__":
    demo.launch()