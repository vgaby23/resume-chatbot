from .config import *
from langchain.agents import create_agent
from .tools import find_candidate, evaluate_job_fit, get_applicant_summary, extract_candidate_info

candidate_evaluation_agent = create_agent(
    model= llm,
    tools = [evaluate_job_fit],
    system_prompt = EVALUATION_PROMPT,
    name = 'evaluation_agent',
)

candidate_info_agent = create_agent(
    model= llm,
    tools = [find_candidate, extract_candidate_info],
    system_prompt = CANDIDATE_INFO_PROMPT,
    name = 'candidate_info_agent',
)

overview_agent = create_agent(
    model= llm,
    tools = [get_applicant_summary],
    system_prompt = OVERVIEW_PROMPT,
    name = 'overview_agent',
)