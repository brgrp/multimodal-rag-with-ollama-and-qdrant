import logging

import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)


class TextEmbedder:
    def __init__(self, model_name, token=None):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.model = AutoModel.from_pretrained(model_name, token=token)
            logging.info(f"Initialized TextEmbedder with model: {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize TextEmbedder. Error: {e}")
            raise

    def embed_text(self, text):
        """Generate embeddings for given text."""
        logging.info(f"Embedding text: {text[:50]}...")  # Log first 50 characters
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
                logging.info("Generated embeddings")
                return embeddings.numpy()
        except Exception as e:
            logging.error(f"Failed to generate embeddings. Error: {e}")
            raise
            raise
