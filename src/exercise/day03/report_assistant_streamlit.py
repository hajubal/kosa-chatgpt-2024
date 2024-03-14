import streamlit as st
import os
from openai import OpenAI


client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")
threadid = ""
assistantid = "asst_zxHv2BqOLPomlip0CMGlid9L"

def create_generator(history):
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "당신은 불친절한 시스템 관리자 입니다. 반말과 욕설을 포함해서 질문에 대한 대답을 충실하게 합니다."
    #     },
    # ]

    run = client.beta.threads.runs.create(
        thread_id=threadid,
        assistant_id=assistantid,
        instructions=history,
    )

    import time

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(
            thread_id=threadid,
            run_id=run.id
        )
        time.sleep(0.1)

    thread_messages = client.beta.threads.messages.list(threadid)

    return thread_messages.data

messages = st.container(height=300)

if prompt := st.chat_input("Say something"):
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 저장된 메시지 출력
    if 'messages' in st.session_state:
        for message in st.session_state.messages:
            messages.chat_message(message["role"]).write(message["content"])

    response = create_generator(st.session_state.messages)

    messages.chat_message("assistant").write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
