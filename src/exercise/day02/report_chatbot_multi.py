import gradio as gr
import os
import time
import json
from openai import OpenAI

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

response_history = []

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def bot(history):
    user_message = history[-1][0]

    print(len(history), len(history[0]))

    gen = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": json.dumps(response_history)
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )

    history[-1][1] = ""

    while True:
        response = next(gen)
        delta = response.choices[0].delta

        if delta.content is not None:
            history[-1][1] += delta.content
            yield history
        else:
            break

    response_history.append(history[-1][1])


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join("avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="텍스트를 입력하고 엔터를 치세요.",
            container=False,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot], [chatbot]
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)


demo.queue()
demo.launch(server_name='0.0.0.0')

