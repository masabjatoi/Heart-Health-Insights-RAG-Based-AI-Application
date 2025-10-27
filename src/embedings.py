from typing import List, Any
# Assuming this library is installed, otherwise replace with a local text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from sentence_transformers import SentenceTransformer
import numpy as np

# Note: The import below is commented out as it causes circular dependency 
# if this file is imported by data_loader, but it's kept for original context.
# from src.data_loader import load_all_documents 

class DocumentChunk:
    """Mock class to replicate the structure of a chunk returned by LangChain splitters."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"DocumentChunk(content='{self.page_content[:50]}...', metadata={self.metadata})"


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 700, chunk_overlap: int = 250):
        # Adjusted chunk size/overlap slightly to improve context capture around key terms like acronyms
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[DocumentChunk]:
        """
        Splits a list of documents (which should contain a 'page_content' attribute 
        or be a simple string/object that can be processed) into smaller chunks.
        """
        # Note: We use a placeholder for split_documents logic since we don't have the LangChain Document object
        # but rely on the RecursiveCharacterTextSplitter behavior.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # We assume documents are objects with a 'page_content' attribute for splitting
        # If documents are just strings, the call below needs adjustment.
        
        # Mocking document splitting for runnable example:
        all_chunks = []
        for doc in documents:
            content = getattr(doc, 'page_content', doc)
            if isinstance(content, str):
                texts = splitter.split_text(content)
                for text in texts:
                    all_chunks.append(DocumentChunk(page_content=text, metadata=getattr(doc, 'metadata', {})))

        print(f"[INFO] Split {len(documents)} documents into {len(all_chunks)} chunks.")
        return all_chunks

    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generates embeddings for a list of document chunks."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage (requires data_loader.py and a 'data' directory)
if __name__ == "__main__":
    # Mocking a load function for local test
    def load_all_documents(path):
        return [
            "Healthcare Data Prediction (HDP) is a critical area. It often involves complex features. "
            "Convolutional Neural Networks (CNN) are powerful in feature extraction from images, but can be adapted to time-series data.",
            "Support Vector Machines (SVM) are excellent for classification tasks and work well with high-dimensional data, "
            "making them complementary to CNN outputs in HDP pipelines. HDP accuracy relies on clean features."
        ]
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example chunk:", chunks[0] if len(chunks) > 0 else None)
    print("[INFO] Example embedding shape:", embeddings[0].shape if len(embeddings) > 0 else None)
