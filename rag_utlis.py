import os
import pandas as pd
import pathlib
from more_itertools import batched
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os


# Configure Gemini
GEMINI_API_KEY = 'API Key'
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")


# Load your local CSV files into one DataFrame
data_path = "C:/Users/saivi/OneDrive/Desktop/Data Science/Projects/AI/RAG/RAG_Project_CarReview/carfax_app/car_data/"
csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

df = pd.DataFrame()
for file in csv_files:
    df_temp = pd.read_csv(os.path.join(data_path, file))
    df = pd.concat([df, df_temp], ignore_index=True)

# Example: split MPG into city/highway
if "MPG" in df.columns:
    df["MPG_City"] = df["MPG"].str.split("/").str[0]
    df["MPG_Highway"] = df["MPG"].str.split("/").str[1]
    df.drop(columns=["MPG"], inplace=True)

df["price_actual"] = df["price"].str.extract(r'([$]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', expand=False)

# Extract MSRP (if present) — second match before 'MSRP'
df["msrp"] = df["price"].str.extract(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*MSRP', expand=False)

# Clean dollar signs and commas, convert to float
df["price_actual"] = df["price_actual"].str.replace("[$,]", "", regex=True).astype(float)
df["msrp"] = df["msrp"].str.replace(",", "", regex=True).astype(float)
df.drop(columns=["price"], inplace=True)



# Prepare for Chroma ingestion
def data_dict(df):
    ids = [f"Review {i}" for i in range(len(df))]
    documents = df["description"].astype(str).tolist()
    metadatas = df.drop(columns=["description"], errors="ignore").to_dict(orient="records")
    return {"ids": ids, "documents": documents, "metadatas": metadatas}

dfdict = data_dict(df)


# STEP 4: Build ChromaDB collection

def build_chroma_collection(
    chroma_path: str,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine"
):
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )
    # delete collection if exists already
    if collection_name in [col.name for col in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name}
    )

    for batch in batched(range(len(documents)), 166):
        start_idx, end_idx = batch[0], batch[-1] + 1
        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx]
        )

    return collection


# Run the build
chroma_path = "C:/Users/saivi/OneDrive/Desktop/Data Science/Projects/AI/RAG/RAG_Project_CarReview/carfax_app/chroma"
embedding_func_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
collection_name = "car_reviews"

collection = build_chroma_collection(
    chroma_path,
    collection_name,
    embedding_func_name,
    dfdict["ids"],
    dfdict["documents"],
    dfdict["metadatas"]
)


# Now test the collection
client = chromadb.PersistentClient(chroma_path)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=embedding_func_name
    )
collection = client.get_collection(
    name = collection_name,
    embedding_function = embedding_func

)


# 1. Search by Price
price_20K = collection.query(
    query_texts=["Show me best cars under Price : 20000"],
    where = {"price_actual": {"$lte": 20000}},
    n_results = 5,
    include = ["documents", "distances", "metadatas"]
)



import re
def ask_gemini_with_rag(user_query, collection, top_k=5):
    match = re.search(r'\$?(\d{1,3}(?:,\d{3})*)', user_query)
    if match:
        match = match.group(1).replace(',', '')
        match = float(match)
        results = collection.query(
            query_texts=[user_query],
            where = {"price_actual": {"$lte": match}},
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

    context = ""
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        context += f"Car {i}:\nDescription: {doc}\n"
        context += "\n".join([f"{k}: {v}" for k, v in meta.items()])
        context += "\n\n"

    print("Context:\n")
    print(context)
    final_prompt = f"""
You are a data analyst providing a market insights report to car dealership executives.

Based on the following data of used cars, generate a detailed analysis covering:

1. Pricing trends and regional patterns.
2. Fuel efficiency comparisons.
3. Accident history and vehicle reliability.
4. Recommendations for best value purchases.
5. Suggested sales strategy for budget-conscious buyers.


Context:
{context}

Question: {user_query}

Answer:"""

    response = gemini_model.generate_content(final_prompt)
    return response.text




