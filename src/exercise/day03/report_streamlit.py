import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
import os
import openai
from langchain.vectorstores import FAISS

openai.api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource
def loading():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local("faiss-nj-pdf-data", embeddings=embeddings, allow_dangerous_deserialization=True)

def create_generator(text):
    content = "당신은 친절한 어시스턴트입니다. 주어진 데이터를 보고 사용자에 친절하게 대답하세요.\n"
    content += "*" * 50
    db = loading()
    docs = db.similarity_search(text)

    for doc in docs:
        content += doc.page_content + "\n"
        content += "*" * 50

    messages = [
        {
            "role": "system",
            "content": content
        },
        {
            "role": "user",
            "content": text
        },
    ]

    gen = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True
    )
    return gen

messages = st.container(height=300)

if prompt := st.chat_input("Say something"):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append(["user", prompt])

    # 저장된 메시지 출력
    if 'messages' in st.session_state:
        for message in st.session_state.messages:
            messages.chat_message(message[0]).write(message[1])

    gen = create_generator(prompt)

    result = messages.chat_message("assistant").write_stream(gen)

    st.session_state.messages.append(["assistant", result])
