import os
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader , PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Currently only supports PDF files.
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files (Deactivated)
    # txt_files = list(data_path.glob('**/*.txt'))
    # print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    # for txt_file in txt_files:
    #     print(f"[DEBUG] Loading TXT: {txt_file}")
    #     try:
    #         loader = TextLoader(str(txt_file))
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files (Deactivated)
    # csv_files = list(data_path.glob('**/*.csv'))
    # print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    # for csv_file in csv_files:
    #     print(f"[DEBUG] Loading CSV: {csv_file}")
    #     try:
    #         loader = CSVLoader(str(csv_file))
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files (Deactivated)
    # xlsx_files = list(data_path.glob('**/*.xlsx'))
    # print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    # for xlsx_file in xlsx_files:
    #     print(f"[DEBUG] Loading Excel: {xlsx_file}")
    #     try:
    #         # Note: UnstructuredExcelLoader requires the `unstructured` library.
    #         loader = UnstructuredExcelLoader(str(xlsx_file))
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # Word files (Deactivated)
    # docx_files = list(data_path.glob('**/*.docx'))
    # print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    # for docx_file in docx_files:
    #     print(f"[DEBUG] Loading Word: {docx_file}")
    #     try:
    #         # Note: Docx2txtLoader requires the `docx2txt` library.
    #         loader = Docx2txtLoader(str(docx_file))
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # JSON files (Deactivated)
    # json_files = list(data_path.glob('**/*.json'))
    # print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    # for json_file in json_files:
    #     print(f"[DEBUG] Loading JSON: {json_file}")
    #     try:
    #         # Note: JSONLoader requires the `jq` library and a valid schema.
    #         loader = JSONLoader(
    #             file_path=str(json_file), 
    #             jq_schema='.'  
    #         )
    #         loaded = loader.load()
    #         print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
    #         documents.extend(loaded)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to load JSON {json_file}: {e}. "
    #               f"Ensure 'jq' is installed and 'jq_schema' matches your JSON structure.")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    # Ensure the data directory exists so the globbing doesn't fail
    os.makedirs("data", exist_ok=True)
    
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
