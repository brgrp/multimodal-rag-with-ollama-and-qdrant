import logging
import os
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)


class DocumentProcessor:
    def __init__(self, documents_dir: str, embedder: Any):
        self.documents_dir = documents_dir
        self.embedder = embedder
        self.document_titles = []
        self.document_embeddings = []
        logging.info(f"Initialized DocumentProcessor with directory: {documents_dir}")

    def process_documents(self):
        logging.info("Processing documents...")
        try:
            for filename in tqdm(
                os.listdir(self.documents_dir), desc="Files Processed"
            ):
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.documents_dir, filename)
                    logging.info(f"Reading file: {file_path}")
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        logging.info(f"Embedding content from: {filename}")
                        embedding = self.embedder.embed_text(content)
                        self.document_embeddings.append(embedding)
                        self.document_titles.append(filename)

            if not self.document_embeddings:
                logging.error("No document embeddings were created.")
                raise ValueError(
                    "No document embeddings were created. Please check your documents directory."
                )
        except Exception as e:
            logging.error(f"Failed to process documents. Error: {e}")
            raise
