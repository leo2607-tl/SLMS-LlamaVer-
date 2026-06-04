import os 
from typing import List, Optional
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from dotenv import load_dotenv
load_dotenv()
from config.config import DatabaseConfig
from database.elasticsearch_database import ElasticSearchDB
from database.qdrant_database import QdrantDB
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import set_global_tokenizer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from transformers import AutoTokenizer

HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")

class Retriever(BaseRetriever):
    def __init__(
        self, 
        embed_model: Optional[HuggingFaceEmbedding] = None, 
        reranker: Optional[LLMRerank] = None
    ):
        super().__init__()
        
        self.dbcf = DatabaseConfig()
        self.qdrant = QdrantDB(url = self.dbcf.QDRANT_URL)
        self.es = ElasticSearchDB(url = self.dbcf.ELASTIC_SEARCH_URL)
        
        self.embed_model = embed_model
        Settings.embed_model = self.embed_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.dbcf.HF_TOKENIZER_MODEL,
            trust_remote_code=True,
            cache_dir=self.dbcf.CACHE_DIR
        )
        set_global_tokenizer(self.tokenizer.encode)
        
        self.vector_store = QdrantVectorStore(
            client = self.qdrant.qr_client,
            collection_name = self.dbcf.QDRANT_COLLECTION,
        )

        storage_context = StorageContext.from_defaults(vector_store = self.vector_store)
        self.semantic_retriever = VectorStoreIndex.from_vector_store(
            vector_store = self.vector_store,
            storage_context = storage_context
        ).as_retriever(similarity_top_k = self.dbcf.TOP_K)

        self.reranker = reranker

    def _keywords_search(
        self, 
        query: str
    ) -> List[NodeWithScore]:
        
        from llama_index.core.schema import TextNode
        
        bm25_results = self.es.search(query, self.dbcf.TOP_K)
        
        node_with_scores = []
        for result in bm25_results:
            text_node = TextNode(
                text=result['text'],
                id_=result['doc_id']  
            )
            
            node_with_scores.append(
                NodeWithScore(node=text_node, score=result['score'])
            )
        
        return node_with_scores

    def _semantic_search(
        self,
        query: str
    )-> List[NodeWithScore]:
        
        return self.semantic_retriever.retrieve(query)
    
    def _combine_results_rrf(
        self,
        semantic_results: List[NodeWithScore],
        keyword_results: List[NodeWithScore],
        k: int = 10
    ) -> List[NodeWithScore]:
        
        rrf_scores = {}
        all_nodes = {} 
        
        for rank, node_with_score in enumerate(semantic_results):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (k + rank + 1)
            
            if node_id not in all_nodes:
                all_nodes[node_id] = node_with_score.node
        
        for rank, node_with_score in enumerate(keyword_results):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (k + rank + 1)
            
            if node_id not in all_nodes:
                all_nodes[node_id] = node_with_score.node
        
        combined = [
            NodeWithScore(node=all_nodes[node_id], score=score)
            for node_id, score in rrf_scores.items()
        ]
        
        return sorted(combined, key=lambda x: x.score, reverse=True)[:self.dbcf.TOP_K]


    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        
        try:
            print(f"\n[DEBUG] Reranking {len(nodes)} nodes for query: {query[:50]}...")
            
            if not nodes:
                print("[WARNING] No nodes to rerank, returning empty list")
                return []
            
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes=nodes,
                query_str=query,
            )
            
            print(f"[DEBUG] Reranker returned {len(reranked_nodes)} nodes")
            
            if not reranked_nodes:
                print("[WARNING] Reranker returned empty results, using original nodes")
                return nodes[:self.dbcf.TOP_K]
            
            return reranked_nodes
            
        except Exception as e:
            print(f"[ERROR] Reranking failed: {str(e)}")
            print(f"[INFO] Returning original nodes without reranking")
            return nodes[:self.dbcf.TOP_K]


    def _retrieve(
        self,
        query: str
    ) -> List[NodeWithScore]:
        print("[DEBUG] Retrieving...")
        semantic_results = self._semantic_search(query)
        keyword_results = self._keywords_search(query)
        print("[DEBUG] combine results...")
        combined_results = self._combine_results_rrf(
            semantic_results, 
            keyword_results
        )
        print("[DEBUG] rerank results...")
        reranker_results = self.rerank(
            query,
            combined_results
        )
        print("[DEBUG] Retrieval complete.")
        return reranker_results