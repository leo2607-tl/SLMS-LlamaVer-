import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()
from RAG.query_engine import QueryEngine

class Synthesizer:
    def __init__(
        self,
        synthesizer,
        retriever
    ):

        self.synthesizer = synthesizer
        self.retriever = retriever
        
        self.query_engine = QueryEngine(
            retriever = self.retriever,
            synthesizer = self.synthesizer,
        )
        
    def _synthesize(
        self,
        query: str
    ) -> str:
        
        response_obj = self.query_engine.custom_query(query)
        print("Response in syn:", response_obj)
        return response_obj.response