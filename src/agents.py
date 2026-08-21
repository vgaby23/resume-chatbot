from .config import *
from langchain.agents import create_agent
from .tools import tools_info, tools_evaluation, tools_overview

candidate_evaluation_agent = create_agent(
    model= llm,
    tools = tools_evaluation,
    system_prompt = EVALUATION_PROMPT,
    name = 'candidate_evaluation_agent',
)

candidate_info_agent = create_agent(
    model= llm,
    tools = tools_info,
    system_prompt = CANDIDATE_INFO_PROMPT,
    name = 'candidate_info_agent',
)

overview_agent = create_agent(
    model= llm,
    tools = tools_overview,
    system_prompt = OVERVIEW_PROMPT,
    name = 'overview_agent',
)
