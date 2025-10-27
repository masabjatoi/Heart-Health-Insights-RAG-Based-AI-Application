from fastapi import FastAPI
from pydantic import BaseModel

# Assuming RAGSearch and src.search exist in your environment
from src.search import RAGSearch 

app = FastAPI()
# Initialize the RAG search system
# In a real application, you should handle RAGSearch initialization errors here
rag = RAGSearch()

class Query(BaseModel):
    # Changed 'question' to 'query' to match the frontend's json payload structure
    query: str 

# This endpoint is called by the Streamlit frontend
@app.post("/rag/search")
def search_query(data: Query):
    # The frontend expects a key named "summary"
    # Access the query using data.query
    response = rag.search_and_summarize(data.query, top_k=3)
    return {"summary": response}

@app.get("/")
def home():
    # Health check endpoint for the frontend to verify connectivity
    return {"status": "RAG API is running 🚀"}


if __name__ == "__main__":
    import uvicorn
    # The frontend code calls http://127.0.0.1:8000/, so the port must match
    uvicorn.run(app, host="0.0.0.0", port=8000)
