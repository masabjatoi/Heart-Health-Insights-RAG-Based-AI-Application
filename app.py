import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data_loader import load_all_documents
from src.embedings import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import uvicorn
import time

app = FastAPI(title="RAG Backend API")
store = FaissVectorStore("faiss_store")
rag_search = None  # Will initialize after vectorstore is ready

class QueryRequest(BaseModel):
    query: str

# --- Startup event: build/load FAISS and init RAG ---
@app.on_event("startup")
def startup_event():
    global rag_search
    faiss_index_path = os.path.join("faiss_store", "faiss.index")

    if not os.path.exists(faiss_index_path):
        print("[INFO] FAISS index not found. Building vector store...")
        docs = load_all_documents("data")
        if not docs:
            print("[WARNING] No documents found in 'data' folder.")
        else:
            store.build_from_documents(docs)
    else:
        store.load()

    rag_search = RAGSearch(vectorstore=store)
    print("[INFO] RAGSearch initialized successfully!")

# --- Endpoint ---
@app.post("/rag/search")
def rag_query(request: QueryRequest):
    if rag_search is None:
        raise HTTPException(status_code=503, detail="RAGSearch not initialized yet.")

    try:
        summary = rag_search.search_and_summarize(request.query, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search error: {e}")

    return {"query": request.query, "summary": summary}

# --- Health check ---
@app.get("/")
def home():
    return {"status": "RAG backend is running 🚀"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
