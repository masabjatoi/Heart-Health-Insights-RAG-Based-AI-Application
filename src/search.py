# src/search.py
import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        vectorstore: FaissVectorStore = None,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        # Use provided vectorstore or create a new one
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
            faiss_path = os.path.join(persist_dir, "faiss.index")
            meta_path = os.path.join(persist_dir, "metadata.pkl")
            if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
                from src.data_loader import load_all_documents
                docs = load_all_documents("data")
                if docs:
                    self.vectorstore.build_from_documents(docs)
            else:
                self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            print("[WARNING] GROQ_API_KEY not set. LLM calls may fail.")

        self.llm = ChatGroq(model=llm_model, temperature=0.1, max_tokens=1024)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Retrieve documents from vectorstore
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        if not texts:
            return "No relevant documents found."

        # Join top-k retrieved documents
        context = "\n\n".join(texts[:top_k])  # optional: limit chars per doc if needed

        # Prompt instructs LLM to give only a concise answer
        prompt = f"""
You are a knowledgeable and reliable medical assistant specializing in heart diseases and general health conditions. 
Use the provided context to accurately answer the user's question. 
Focus on medical accuracy, clarity, and relevance to the question. 
If the question is about heart disease, provide detailed and expert-level insights on causes, symptoms, treatment, prevention, and lifestyle recommendations. 
If it concerns another disease, use the context to give a clear, concise, and medically sound response.

Do NOT copy the text from the context verbatim—summarize and rephrase it naturally in an informative, human-like tone. 
Only include information that is supported by the context. Avoid adding assumptions or unverified details.

Question: {query}

Context: {context}

Answer:
"""


        response = self.llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()


# --- Example usage ---
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "what is  HDP ?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
