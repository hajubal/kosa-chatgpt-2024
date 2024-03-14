import gradio as gr
import os
from openai import OpenAI
import report_db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
index = report_db.loadPdfToDB()

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def bot(history):
    user_message = history[-1][0]

    retriever = index.vectorstore.as_retriever()

    template = """다음 지문에만 근거해서 질문에 답하세요:
    {context}

    질문: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI()
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(  # 여러개의 Runnable을 병렬적으로 실행
        {"context": retriever, "question": RunnablePassthrough()}  # RunnablePassthrough는 값을 입력받아 그대로 전달하는 객체
    )
    chain = setup_and_retrieval | prompt | model | output_parser

    history[-1][1] = ""

    for chunk in chain.stream(user_message):
        history[-1][1] += chunk
        yield history


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

