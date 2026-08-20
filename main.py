from src.config import *
from src.agents import *
from dotenv import load_dotenv
import os 
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langgraph_supervisor import create_supervisor
import streamlit as st

load_dotenv()


def resume_review_supervisor():
    workflow = create_supervisor(
        agents = [candidate_info_agent, candidate_evaluation_agent, candidate_search_agent],
        model = llm,
        prompt = SUPERVISOR_PROMPT,
    )

    supervisor = workflow.compile()

    return supervisor

if __name__ == "__main__":
    supervisor = resume_review_supervisor()

    # streamlit interface
    if os.getenv('OPENAI_API_KEY'):
        supervisor = resume_review_supervisor()
        if supervisor:
            print('Supervisor created successfully.')
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # create chat interface 
        for messages in st.session_state['messages']:
            with st.chat_message(messages['role']):
                st.markdown(messages['content'])

        if prompt_U := st.chat_input("Hi! I'm Sunny HR Assistant. How can I help you today?"):
            with st.chat_message("user"):
                st.markdown(prompt_U)

            st.session_state.messages.append({"role": "user", "content": prompt_U})

            with st.chat_message("assistant"):
                print(supervisor)
                final_prompt = f'''User question: {prompt_U}
                History:\n{st.session_state['messages']}
                '''
                response= supervisor.invoke({"messages": [{"role": "user", "content": final_prompt}]})
                answer = response["messages"][-1].content
                st.markdown(answer)


            st.session_state.messages.append({"role": "assistant", "content": answer})