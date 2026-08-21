import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os

st.set_page_config(page_title="Sunny — HR Intelligence", page_icon="🤖")

api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please enter your API key in the sidebar or setup secrets to continue.")
    st.stop()

from langgraph_supervisor import create_supervisor
from src.config import llm, SUPERVISOR_PROMPT
from src.agents import candidate_info_agent, candidate_evaluation_agent, overview_agent

# generate multiagent supervisor
@st.cache_resource
def get_supervisor():
    return create_supervisor(
        agents=[candidate_info_agent, candidate_evaluation_agent, overview_agent],
        model=llm,
        prompt=SUPERVISOR_PROMPT,
    ).compile()

# streamlit interface
supervisor = get_supervisor()

# streamlit UI setup
st.title("🤖 Sunny — HR Intelligence Assistant")
st.caption("Streamline candidate search, applicant analytics, and resume evaluations.")
st.divider()

# message history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous message
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant"):
            with st.expander(f"Tool Calls: ({msg.name})", expanded=False):
                st.code(msg.content, language="json")
    elif isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("Transferring"):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# user prompt
if prompt := st.chat_input("Chat with Sunny!"):
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    st.session_state.chat_history.append(user_message)   

    with st.chat_message("user"):
        st.markdown(user_message.content)

    with st.chat_message("assistant"):
        with st.status("Sunny is thinking...", expanded=False):
            response = supervisor.invoke({"messages": st.session_state.messages})

        answer = response["messages"][-1].content
        st.markdown(answer)
        st.session_state.chat_history.append(AIMessage(content=answer))

        # displaying tool call 
        for msg in response["messages"][len(st.session_state.messages):]:
            if isinstance(msg, ToolMessage):
                with st.expander(f"Tool Call: {getattr(msg, 'name', 'Tool')}", expanded=False):
                    st.markdown(msg.content)

        # Displaying tool usage
        total_input = 0
        total_output = 0
        total_tokens = 0

        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                usage = msg.usage_metadata

                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
        with st.expander('LLM Usage', expanded=False):
            st.write(f"Input tokens: {total_input}")
            st.write(f"Output tokens: {total_output}")
            st.write(f"Total tokens: {total_tokens}")

        # Update full session history
        st.session_state.messages = response["messages"]
