import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from RAG.retriever import Retriever
from llama_index.llms.vllm import Vllm

from RAG.synthesizer import Synthesizer
from RAG.retriever import Retriever
from RAG.prompt import SYNTHESIZE_PROMPT, COMPACT_AND_REFINE_PROMPT
from llama_index.core import PromptTemplate, Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
os.environ["HF_HOME"] = "/workspace/aiclub02/leo/.cache/vLLM/models/"

llm = Vllm(
            model="meta-llama/Llama-3.1-8B",
            dtype="float16",
            tensor_parallel_size=2,
            temperature=0,
            max_new_tokens=1000,
            vllm_kwargs={
                "swap_space": 1,
                "gpu_memory_utilization": 0.6,
                "max_model_len": 4096,
            },
        )
        
Settings.llm = llm

synthesizer = get_response_synthesizer(
    llm = Settings.llm,
    response_mode=ResponseMode.COMPACT,
    text_qa_template=PromptTemplate(SYNTHESIZE_PROMPT),
    refine_template=PromptTemplate(COMPACT_AND_REFINE_PROMPT),
)

embed_model = HuggingFaceEmbedding(
    model_name="AITeamVN/Vietnamese_Embedding",
    cache_folder = "/workspace/aiclub02/leo/.cache/huggingface" 
)

reranker = LLMRerank(
    llm=llm,
    top_n=7,
    choice_batch_size=5
)

retriever = Retriever(
    embed_model=embed_model,
    reranker=reranker
)

Resp = Synthesizer(
    synthesizer=synthesizer,
    retriever=retriever
)

query = "hướng dẫn tôi Tạo hoạt động Feedback"
response = Resp._synthesize(query)
print("Final Response:", response)
