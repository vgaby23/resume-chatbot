# from src.config import *
# from src.agents import *
# from dotenv import load_dotenv
# import os 
# from langgraph_supervisor import create_supervisor
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# import streamlit as st

# st.set_page_config(
#     page_title="Sunny | HR Assistant",  
#     page_icon="🤖",                     
#     layout="centered"                   
# )

# load_dotenv()

# @st.cache_resource
# def resume_review_supervisor():
#     workflow = create_supervisor(
#         agents = [candidate_info_agent, candidate_evaluation_agent, overview_agent],
#         model = llm,
#         prompt = SUPERVISOR_PROMPT,
#     )

#     supervisor = workflow.compile()

#     return supervisor

# if __name__ == "__main__":
    
#     st.title("🤖 Sunny — HR Intelligence Assistant")
#     st.caption("Streamline candidate search, applicant analytics, and resume evaluations.")
#     st.divider()

#     # streamlit interface
#     if os.getenv('OPENAI_API_KEY'):
#         supervisor = resume_review_supervisor()

#         if 'messages' not in st.session_state:
#             st.session_state['messages'] = []

#         # create chat interface 
#         for messages in st.session_state['messages']:
#             with st.chat_message(messages['role']):
#                 st.markdown(messages['content'])
            
#         # Handle User input
#         if prompt_U := st.chat_input("Chat with Sunny!"):
#             # Display user message
#             with st.chat_message("user"):
#                 st.markdown(prompt_U)

#             # Store user message for UI and graph state
#             st.session_state.messages.append({"role": "user", "content": prompt_U})
#             st.session_state.langchain_messages.append(HumanMessage(content=prompt_U))

#             with st.chat_message("assistant"):
#                 with st.status("Sunny is finding the best information for you...", expanded=False) as status:
#                     final_prompt = f'''User question:{prompt_U}
#                     History:\n{st.session_state.messages}'''
#                     response = supervisor.invoke({'messages': st.session_state.messages})
#                     status.update(label="Complete!", state="complete", expanded=False)

#                 rag_tool_outputs = []
#                 for msg in response["messages"]:
#                     if isinstance(msg, ToolMessage) or getattr(msg, "type", None) == "tool":
#                         tool_name = getattr(msg, "name", "").lower()
#                         if "transfer" not in tool_name:
#                             rag_tool_outputs.append(f"[{msg.name}]\n{msg.content}")

#                 tool_data_str = "\n\n".join(rag_tool_outputs) if rag_tool_outputs else None        
#                 if tool_data_str:
#                     with st.expander("Tool Calls:", expanded=False):
#                         st.code(tool_data_str, language="python")

#                 answer = response["messages"][-1].content
#                 st.markdown(answer)
#                 print("response:", response)

#             st.session_state.messages.append({"role": "assistant", "content": answer})
#             st.session_state.langchain_messages.append(AIMessage(content=answer))

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_supervisor import create_supervisor

from src.config import llm, SUPERVISOR_PROMPT
from src.agents import candidate_info_agent, candidate_evaluation_agent, overview_agent

st.set_page_config(page_title="Sunny | HR Assistant", page_icon="🤖", layout="centered")
load_dotenv()

@st.cache_resource
def get_supervisor():
    return create_supervisor(
        agents=[candidate_info_agent, candidate_evaluation_agent, overview_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
    ).compile()

supervisor = get_supervisor()

st.title("🤖 Sunny — HR Intelligence Assistant")
st.caption("Streamline candidate search, applicant analytics, and resume evaluations.")
st.divider()

# 1. Single source of truth for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render historical messages directly
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage) and "transfer" not in getattr(msg, "name", "").lower():
        with st.chat_message("assistant"):
            with st.expander("Tool Calls:", expanded=False):
                st.code(f"[{msg.name}]\n{msg.content}", language="python")
    elif isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("Transferring"):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 3. Handle user prompt
if prompt := st.chat_input("Chat with Sunny!"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.status("Sunny is thinking...", expanded=False):
            response = supervisor.invoke({"messages": st.session_state.messages})

        # Process only NEW messages returned in this turn
        new_messages = response["messages"][len(st.session_state.messages):]

        for msg in new_messages:
            if isinstance(msg, ToolMessage) and "transfer" not in getattr(msg, "name", "").lower():
                with st.expander("Tool Calls:", expanded=False):
                    st.code(f"[{msg.name}]\n{msg.content}", language="python")
            elif isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("Transferring"):
                st.markdown(msg.content)

        # Update full session history
        st.session_state.messages = response["messages"]