from src.config import *
from utils.data_loader import load_listing
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import CrossEncoder
import pandas as pd
import os

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

df = load_listing()

# Vector database
collection_name = "resume_collection"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-large-v1')

def candidate_search(query: str, top_k: int = 3):

  candidates = qdrant.similarity_search(query, k=top_k *3)
  text = [doc.page_content for doc in candidates]

  ranked = reranker.rank(query, text, return_documents=False, top_k=top_k)

  results = []

  for item in ranked:
    doc = candidates[item["corpus_id"]]
    results.append({
        "candidate_id": doc.metadata["unique_id"],
        "category": doc.metadata['category'],
        "relevance_score": round(float(item["score"]),3),
        "resume": doc.metadata['resume']
    })
    return results

class CandidateEntities(BaseModel):
  years_of_experience: int = Field(
        description="Total years of professional experience. Return 0 if none is found.")
  highest_degree: str = Field(
        description="The highest educational degree earned (e.g., Bachelor of Science, High School Diploma). Return 'Unknown' if not found.")
  hard_skills: list[str] = Field(
        description="A list of technical skills, software, and tools mentioned in the resume.")
  soft_skills: list[str] = Field(
      description= "A list of soft skills such as leadership, time management, etc. that are mentioned in the resume")
  working_experience_history: list[str] = Field(
      description="A list of candidate's past working experience mentioned in the resume in descending order.")
  certification_portfolio:list[str]= Field(
      description="A list of candidate's certification portfolio mentioned in the resume.")

@tool
def find_candidate(query:str) -> str:
  """
  Use this tool when searching for candidate in the vector database based on the users needs, 
  for instance applicants from certain department, or applicants with certain years of experience.
  Always use this tool when the user mentioned the name of the department
  """
  results = candidate_search(query, top_k = 3)
  if not results:
    return "No matched candidates found"
  lines = [f"- {r['candidate_id']} | ({r['category']}) - {r['resume']}" for r in results]
  return "\n".join(lines)

@tool
def get_applicant_summary(category: str) -> str:
    """
    Use this tool to get a statistical summary of the applicants in the database.
    If a specific 'category' is provided, it summarizes that category.
    If no category is provided, it returns an overall summary of all applicants.
    """
    if category:
        subset = df[df['Category'].str.lower() == category.lower()]
        if subset.empty:
            return f"No applicants found for the category: {category}."
        count = len(subset)
        return f"Summary for {category}: {count} total applicants."
    else:
        total_count = len(df)
        category_counts = df['Category'].value_counts().to_dict()
        return f"Overall Summary: {total_count} total applicants. Breakdown by category: {category_counts}"

@tool
def extract_candidate_info(candidate_id: int) -> str:
    """
    Use this tool to extract structured metadata (years of experience, highest degree, soft skills,
    technical skills, working experience history, portfolio and certifications) 
    from a candidate's resume using their unique ID.
    """
    try:
        candidate_id = int(candidate_id)
        subset = df[df['ID'] == candidate_id]
        
        if subset.empty:
            return f"Error: No candidate found with ID '{candidate_id}'."
            
        # Get the raw resume text
        resume_text = subset['Resume_str'].values[0]
        
        structured_llm = llm.with_structured_output(CandidateEntities)
        
        extracted_data = structured_llm.invoke(
            f"Extract the requested information from this resume:\n\n{resume_text}"
        )
        response = extracted_data['messages'][-1].content
        
        return response
        
    except ValueError:
        return "Error: Please provide a valid numeric ID."
    except Exception as e:
        return f"An unexpected error occurred during extraction: {str(e)}"

class JobFitEvaluation(BaseModel):
    match_score_percentage: int = Field(
        description="A score from 0 to 100 indicating how well the candidate's skills and experience match the job description."
    )
    matching_qualifications: list[str] = Field(
        description="A list of key qualifications from the candidate's resume that successfully match the job description."
    )
    missing_requirements: list[str] = Field(
        description="A list of requirements in the job description that appear to be missing from the candidate's resume."
    )
    recommendation_rationale: str = Field(
        description="A brief 2-3 sentence paragraph explaining the overall fit and whether the candidate is recommended for an interview."
    )

@tool
def evaluate_job_fit(candidate_id: int, job_description: str) -> str:
    """
    Use this tool to evaluate how well a specific candidate matches a target job description.
    It returns a match score, matching qualifications, missing requirements, and a recommendation.
    """
    try:
        candidate_id = int(candidate_id)
        subset = df[df['ID'] == candidate_id]
        
        if subset.empty:
            return f"Error: No candidate found with ID '{candidate_id}'."
            
        resume_text = subset['Resume_str'].values[0]
        
        # Bind the Pydantic schema to the LLM
        structured_eval_llm = llm.with_structured_output(JobFitEvaluation)
        
        # Construct the prompt comparing the two texts
        evaluation_prompt = (
            f"Evaluate the provided resume against the given job description.\n\n"
            f"--- JOB DESCRIPTION ---\n{job_description}\n\n"
            f"--- CANDIDATE RESUME ---\n{resume_text}"
        )
        
        # Run the evaluation
        evaluation_result = structured_eval_llm.invoke(evaluation_prompt)

        response = evaluation_result['messages'][-1].content
        
        return response
        
    except ValueError:
        return "Error: Please provide a valid numeric candidate_id."
    except Exception as e:
        return f"An unexpected error occurred during evaluation: {str(e)}"

tools = [find_candidate, get_applicant_summary, extract_candidate_info, evaluate_job_fit]