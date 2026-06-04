import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from qdrant_client import QdrantClient, models
from config.config import DatabaseConfig
from typing import List

class QdrantDB:
    def __init__(
        self,
        url: str,
    ):
        self.dbcf = DatabaseConfig()
        self.collection_name = self.dbcf.QDRANT_COLLECTION
        self.qr_client = QdrantClient(url = url)
        
    def create_collection(
        self, 
        vector_size: int, 
        distance: str = models.Distance.COSINE
    ):

        if not self.qr_client.collection_exists(collection_name=self.collection_name):
            self.qr_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance)
            ) 
            print(f"Collection {self.collection_name} created")
        else:
            print(f"Collection {self.collection_name} already exists")
            
    def delete_collection(
        self    
    ): 
        if self.qr_client.collection_exists(collection_name = self.collection_name):
            self.qr_client.delete_collection(collection_name = self.collection_name)
            print(f"Collection {self.collection_name} deleted")
        else:
            print(f"Collection {self.collection_name} not found")
    
    def add_vectors(
        self,
        vector_ids: List[str],
        vectors: List[List[float]],
        payloads: List[dict]
    ):
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=vid,
                vector=vec,
                payload=payload
            )
            for vid, vec, payload in zip(vector_ids, vectors, payloads)
        ]
        
        operation_info = self.qr_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True  
        )
        
        return operation_info
    
    def get_all_points(
        self, 
        collection_name: str
    ):

        limit = 1000
        offset = 0
        results = []
        while True:
            response = self.qr_client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            points = response[1]
            if not points:
                break
            results.extend(points)
            offset += limit
        return results
    
    def counting(self):
        result = self.qr_client.count(
            collection_name=self.collection_name,
            exact=True 
        )
        return result.count 
