import os
import sys
import re
import uuid
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from typing import List, Optional
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document
from database.elasticsearch_database import ElasticSearchDB
from elasticsearch.helpers import bulk
from database.qdrant_database import QdrantDB
from config.config import DatabaseConfig
from config.schema import Documents

class Ingest:
    def __init__(self):
        
        self.dbcf = DatabaseConfig
        self.qr_client = QdrantDB(url = self.dbcf.QDRANT_URL)
        self.es_client = ElasticSearchDB(url = self.dbcf.ELASTIC_SEARCH_URL)
        self.embedding_model = HuggingFaceEmbedding(
            model_name = self.dbcf.HF_EMBEDDING_MODEL,
            device = self.dbcf.EMBEDDING_DEVICE,
            cache_folder= self.dbcf.CACHE_DIR,
        )
        Settings.embed_model = self.embedding_model
    
    def get_embedding(
        self, 
        text: str
    )-> List[float]:
        
        return self.embedding_model.get_text_embedding(text)
        
    def elasticsearch_ingest(
        self,
        documents: List[Documents]
    ):
        
        actions = [
            {
                '_index': self.dbcf.ELASTIC_SEARCH_INDEX,
                '_id': doc['doc_id'],
                '_source': {
                    'doc_id': doc['doc_id'],
                    'text': doc['text']
                }
            }
            for doc in documents
        ]
        
        success, _ = bulk(self.es_client.es_client, actions)
        return success
    
    def qdrant_ingest(
        self,
        documents: List[Documents]
    ):
        
        vector_ids = [doc['doc_id'] for doc in documents]
        vectors = [self.get_embedding(doc['text']) for doc in documents]
        payloads = [{'doc_id': doc['doc_id'], 'text': doc['text']} for doc in documents]
        
        self.qr_client.add_vectors(
            vector_ids=vector_ids,
            vectors=vectors,
            payloads=payloads
        )
        
        return len(vector_ids)
    
    def _ingest(
        self,
        documents: List[Documents]
    ):
        
        es_success = self.elasticsearch_ingest(documents)
        qr_count = self.qdrant_ingest(documents)
        
        return {
            'elasticsearch': es_success,
            'qdrant': qr_count
        }
        

