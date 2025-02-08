import logging
import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.endpoint import EndPoint
from src.qdrant_index import QdrantIndex
from src.text_embedder import TextEmbedder

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()


class SimpleRAG:
    def __init__(
        self,
        embedder: TextEmbedder,
        processor: DocumentProcessor,
        index: QdrantIndex,
        api_key: str,
        api_url: str,
        top_n: int = 3,
        model: str = "llama3.2",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 3000,
    ):
        """
        Initialize the SimpleRAG class.

        :param embedder: Instance of TextEmbedder
        :param processor: Instance of DocumentProcessor
        :param index: Instance of QdrantIndex
        :param api_key: API key for the endpoint
        :param api_url: API URL for the endpoint
        :param top_n: Number of top documents to retrieve
        :param model: Model to use for generating responses
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling probability
        :param max_tokens: Maximum number of tokens in the response
        """
        self.embedder = embedder
        self.processor = processor
        self.index = index
        self.api_key = api_key
        self.api_url = api_url
        self.top_n = top_n
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        logging.info("Initialized SimpleRAG")

    def _create_payload(self, prompt: str) -> dict:
        """
        Create payload for the request.

        :param prompt: The prompt to send to the API
        :return: Payload dictionary
        """
        logging.info("Creating payload for the request")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant named DocumentFinder",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

    def retrieve(self, query: str) -> Tuple[str, List[str]]:
        """
        Search for relevant documents.

        :param query: The query to search for
        :return: Retrieved content and titles
        """
        logging.info(f"Embedding query: {query}")
        try:
            query_embedding = self.embedder.embed_text(query).flatten()
            logging.info("Searching for top similar documents...")
            retrieved_titles = self.index.search(query_embedding, self.top_n)

            retrieved_content = ""
            for title in retrieved_titles:
                logging.info(f"Retrieving content from document: {title}")
                with open(
                    os.path.join(self.processor.documents_dir, title),
                    "r",
                    encoding="utf-8",
                ) as file:
                    retrieved_content += file.read() + "\n"

            return retrieved_content, retrieved_titles
        except Exception as e:
            logging.error(f"Failed to retrieve documents. Error: {e}")
            raise

    def generate_response(self, query: str, retrieved_content: str) -> Optional[str]:
        """
        Generate a response based on retrieved content.

        :param query: The query to answer
        :param retrieved_content: The content retrieved from documents
        :return: Generated response
        """
        logging.info("Generating response...")
        endpoint = EndPoint(self.api_key, self.api_url)
        prompt = f"Based on the following documents:\n{retrieved_content}\nAnswer the query: {query}"
        try:
            response = endpoint.send(self._create_payload(prompt))
            response_text = response["message"]["content"]
            return response_text
        except Exception as e:
            logging.error(f"Failed to generate response. Error: {e}")
            return None

    def update_documents(self, new_documents_dir: str):
        """
        Update the documents and reprocess them.

        :param new_documents_dir: Directory containing new text files
        """
        logging.info(f"Updating documents from directory: {new_documents_dir}")
        self.processor.documents_dir = new_documents_dir
        try:
            self.processor.process_documents()
        except ValueError as e:
            logging.error(f"Error processing documents: {e}")
            return

        dimension = self.processor.document_embeddings[0].shape[1]
        self.index = QdrantIndex(dimension)
        self.index.add_embeddings(
            self.processor.document_embeddings, self.processor.document_titles
        )


def create_simple_rag(
    documents_dir: str,
    api_key: str,
    api_url: str,
    huggingface_token: str,
    model: str = "llama3.2",
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_tokens: int = 3000,
) -> Optional[SimpleRAG]:
    """
    Create an instance of SimpleRAG.

    :param documents_dir: Directory containing text files
    :param api_key: API key for the endpoint
    :param api_url: API URL for the endpoint
    :param huggingface_token: Token for Hugging Face
    :param model: Model to use for generating responses
    :param temperature: Sampling temperature
    :param top_p: Nucleus sampling probability
    :param max_tokens: Maximum number of tokens in the response
    :return: Instance of SimpleRAG
    """
    logging.info("Creating SimpleRAG instance")
    embedder = TextEmbedder(
        model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
        token=huggingface_token,
    )
    processor = DocumentProcessor(documents_dir=documents_dir, embedder=embedder)

    try:
        processor.process_documents()
    except ValueError as e:
        logging.error(f"Error processing documents: {e}")
        return None

    dimension = processor.document_embeddings[0].shape[1]
    index = QdrantIndex(dimension)
    index.add_embeddings(processor.document_embeddings, processor.document_titles)

    return SimpleRAG(
        embedder=embedder,
        processor=processor,
        index=index,
        api_key=api_key,
        api_url=api_url,
        max_tokens=max_tokens,
    )
