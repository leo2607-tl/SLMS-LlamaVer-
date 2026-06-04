import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config.config import DatabaseConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from RAG.retriever import Retriever
from llama_index.llms.vllm import Vllm

os.environ["HF_HOME"] = "/workspace/aiclub02/leo/.cache/vLLM/models/"

class TestRetriever:
    def __init__(self):
        self.dbcf = DatabaseConfig()
        self.setup_retriever()
        
    def setup_retriever(self):
        print("Đang khởi tạo Retriever...")
        
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.dbcf.HF_EMBEDDING_MODEL,
            cache_folder=self.dbcf.CACHE_DIR
        )
        
        self.llm = Vllm(
            model="meta-llama/Llama-3.1-8B",
            dtype="float16",
            tensor_parallel_size=2,
            temperature=0,
            max_new_tokens=100,
            vllm_kwargs={
                "swap_space": 2,
                "gpu_memory_utilization": 0.7,
                "max_model_len": 4096,
            },
        )
        
        self.reranker = LLMRerank(
            llm=self.llm,
            top_n=self.dbcf.TOP_K,
            choice_batch_size=5
        )
        
        self.retriever = Retriever(
            embed_model=self.embed_model,
            reranker=self.reranker
        )
        
        print("Retriever đã được khởi tạo thành công\n")

    def test_full_retrieval(self, query: str):
        print(f"\n=== TEST FULL RETRIEVAL PIPELINE ===")
        print(f"Query: {query}\n")
        
        try:
            results = self.retriever._retrieve(query)
            
            print(f"Số kết quả cuối cùng: {len(results)}")
            for i, node_score in enumerate(results, 1):
                print(f"\n--- Kết quả {i} (Sau Reranking) ---")
                print(f"Final Score: {node_score.score:.4f}")
                print(f"Node ID: {node_score.node.node_id}")
                print(f"Text: {node_score.node.text[:300]}...")
            
            return results
        except Exception as e:
            print(f"Lỗi trong full retrieval: {str(e)}")
            return []
    
def main():
    tester = TestRetriever()
    
    sample_query = "cuiz"
    tester.test_full_retrieval(sample_query)
if __name__ == "__main__":
    main()
