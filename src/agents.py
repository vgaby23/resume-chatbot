from .config import *
from langchain.agents import create_agent
from .tools import find_candidate, evaluate_job_fit, extract_candidate_info, get_applicant_summary

candidate_evaluation_agent = create_agent(
    model= llm,
    tools = evaluate_job_fit,
    system_prompt = EVALUATION_PROMPT,
    name = 'evaluator_agent',
)

candidate_info_agent = create_agent(
    model= llm,
    tools = [extract_candidate_info, get_applicant_summary],
    system_prompt = INFO_PROMPT,
    name = 'candidate_info_agent',
)

candidate_search_agent = create_agent(
    model= llm,
    tools = find_candidate,
    system_prompt = CANDIDATE_SEARCH_PROMPT,
    name = 'candidate_search_agent',
)