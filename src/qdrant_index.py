import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO)


class QdrantIndex:
    def __init__(self, dimension: int, collection_name: str = "documents"):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logging.info(
            f"Created Qdrant collection: {self.collection_name} with dimension: {dimension}"
        )

    def add_embeddings(self, embeddings: List[List[float]], titles: List[str]):
        points = [
            PointStruct(
                id=i, vector=embedding.flatten().tolist(), payload={"title": title}
            )
            for i, (embedding, title) in enumerate(zip(embeddings, titles))
        ]
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logging.info(
                f"Added {len(points)} embeddings to collection: {self.collection_name}"
            )
        except Exception as e:
            logging.error(f"Failed to add embeddings. Error: {e}")
            raise

    def search(self, query_embedding: List[float], top_n: int) -> List[str]:
        logging.info(f"Searching for top {top_n} similar documents")
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.flatten().tolist(),
                limit=top_n,
            )
            return [hit.payload["title"] for hit in search_result]
        except Exception as e:
            logging.error(f"Failed to search documents. Error: {e}")
            raise
