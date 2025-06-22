# agent_utils.py
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from crewai import Crew, Agent, Task, Process, LLM
from crewai.tools import BaseTool
from typing import Any
import google.generativeai as genai

# Gemini API key setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Global dict to store context
last_rag_context = {"content": ""}

def initialize_llm():
    return LLM(
        model="gemini/gemini-2.0-flash",
        provider="google",
        api_key=GEMINI_API_KEY
    )


def format_agent_output(raw_output: str) -> str:
    formatting_prompt = f"""
You are a professional report formatter. Clean and structure the following raw response from an AI agent into a well-formatted, bullet-pointed or sectioned summary.

Avoid repeating irrelevant context, and structure it in markdown with:
- Headings for sections (like Vehicle Summary, Market Insight, Fuel Efficiency)
- Bullet points for details
- Short and readable format

Important:
- Do NOT invent or hallucinate any vehicles. Do NOT use prior knowledge. 
ONLY analyze and summarize the cars in context.
- If the context is not relevant, use your knowledge to provide a general summary.

Raw output:
\"\"\"
{raw_output}
\"\"\"

Formatted summary:
"""
    llm_model = initialize_genai_model()
    response = llm_model.generate_content(formatting_prompt)
    print("Formatted Response:", response.text)  # Debugging line
    return response.text



def initialize_genai_model():
    return genai.GenerativeModel(model_name="gemini-2.0-flash")

class RAGRetriever(BaseTool):
    name: str = "RAGRetriever"
    description: str = "Retrieves relevant cars from ChromaDB based on user queries."

    def _run(self, query: str) -> Any:
        print("\n🔧 Tool Selected: RAGRetriever 🔧")
        context = self.retrieve_context_only(query)
        return f"""
    ONLY use the context below to answer the user query. Do NOT add new vehicles or prior knowledge.

    Context:
    \"\"\"{context}\"\"\"
    """

    def retrieve_context_only(self, user_query: str, top_k: int = 5) -> str:
        chroma_path = "C:/Users/saivi/OneDrive/Desktop/Data Science/Projects/AI/RAG/RAG_Project_CarReview/carfax_app/chroma/"
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )
        client = chromadb.PersistentClient(chroma_path)
        collection = client.get_collection(name="car_reviews", embedding_function=embedding_func)

        match = re.search(r'\$?(\d{1,3}(?:,\d{3})*)', user_query)
        if match:
            price = float(match.group(1).replace(',', ''))
            results = collection.query(
                query_texts=[user_query],
                where={"price_actual": {"$lte": price}},
                n_results=top_k,
                include=["documents", "metadatas"]
            )
        else:
            results = collection.query(
                query_texts=[user_query],
                n_results=top_k,
                include=["documents", "metadatas"]
            )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            last_rag_context["content"] = "❌ No relevant cars found in the database for your query."
            return last_rag_context["content"]

        context = ""
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            context += f"Car {i}:\nDescription: {doc}\n"
            context += "\n".join([f"{k}: {v}" for k, v in meta.items()])
            context += "\n\n"

        last_rag_context["content"] = context
        return context

class KnowledgeLLMTool(BaseTool):
    name: str = "KnowledgeLLMTool"
    description: str = "Uses Gemini to answer automotive queries not requiring DB search."

    def _run(self, query: str) -> Any:
        print("\n🔧 Tool Selected: KnowledgeLLMTool 🔧")
        # Clear context to avoid showing old RAG results
        last_rag_context["content"] = "ℹ️ No database context used. This is a general knowledge answer."
        llm_model = initialize_genai_model()
        return llm_model.generate_content(query).text

# Agent definition
llm_model = initialize_llm()
carfax_agent = Agent(
    role="Care Market Intelligence Agent",
    goal="Help users analyze car listings and market trends using only RAG output or either knowledge reasoning. Don't use both at the same time.",
    backstory="You are a smart automotive AI advisor for a dealership.",
    llm=llm_model,
    tools=[RAGRetriever(), KnowledgeLLMTool()],
    verbose=True
)

# Crew setup
crew = Crew(agents=[carfax_agent], tasks=[], process=Process.sequential)

__all__ = ["crew", "Task", "carfax_agent", "last_rag_context", "initialize_genai_model", "format_agent_output"]