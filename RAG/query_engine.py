import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer

class QueryEngine(CustomQueryEngine):
    
    retriever: BaseRetriever
    synthesizer: BaseSynthesizer
    
    def custom_query(
        self,
        query: str
    ):
        
        nodes = self.retriever._retrieve(query)
        response = self.synthesizer.synthesize(query, nodes)
        print("Response in QE:", response)
        return response