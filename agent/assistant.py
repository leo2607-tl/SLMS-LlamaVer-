import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from RAG.retriever import Retriever
from RAG.synthesizer import Synthesizer
from config.config import DatabaseConfig 
from RAG.prompt import SYNTHESIZE_PROMPT, COMPACT_AND_REFINE_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import Settings
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from RAG.prompt import QUESTION_GENERATION_PROMPT
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.vllm import Vllm

HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
os.environ["HF_HOME"] = "/workspace/aiclub02/leo/.cache/vLLM/models/"

class LMS_Assistant:
    def __init__(self, llm):
        self.dbcf = DatabaseConfig()
        self.tools = self.load_tools(llm)
        self.agents = self.load_agent(llm)
        
    def load_tools(self, llm):
        Settings.llm = llm
        
        emb_model = HuggingFaceEmbedding(
            model_name= self.dbcf.HF_EMBEDDING_MODEL,
            cache_folder= self.dbcf.CACHE_DIR
        )
        
        reranker = LLMRerank(
            llm=llm,
            top_n=2,
            choice_batch_size=3
        )
        
        retriever = Retriever(
            embed_model = emb_model,
            reranker = reranker
        )
        
        synthesizer = get_response_synthesizer(
            llm = llm,
            response_mode = ResponseMode.COMPACT,
            text_qa_template = PromptTemplate(SYNTHESIZE_PROMPT),
            refine_template = PromptTemplate(COMPACT_AND_REFINE_PROMPT),
        )
        
        engine = Synthesizer(
            retriever = retriever,
            synthesizer = synthesizer,
        )
        
        subengine= QueryEngineTool(
            query_engine = engine,
            metadata = ToolMetadata(
                name="subqengine",
                description="Use this tool to answer complex questions by breaking them down into simpler sub-questions. Input should be a single question string.",
            )
        )   
        
        qengine = SubQuestionQueryEngine(
            question_gen = PromptTemplate(QUESTION_GENERATION_PROMPT),
            response_synthesizer = synthesizer,
            query_engine_tools = [subengine],
            verbose = True,
            use_async = True
        )
        
        async def en_answer_query(query: str) -> str:
            response = engine._synthesize(query)
            return response
        
        async def sub_answer_query(query: str) -> str:
            response = qengine.query(query)
            return response
        
        en_tool = FunctionTool.from_defaults(
            async_fn = en_answer_query,
            name = "Contextual",
            return_direct = True,
            description = ("Use this tool for simple how-to questions that can be answered in one step using Retrieval Augmentation Generation (RAG). For example, questions asking how to do something")
        )
        
        sub_tool = FunctionTool.from_defaults(
            async_fn = sub_answer_query,
            name = "Decomposition",
            return_direct = True,
            description = ("Use this tool for complex, multi-part, or high-level questions that require breaking them down into smaller questions. Useful when the answer requires answering multiple ideas in succession.")
        )
        
        return [en_tool, sub_tool]
    
    def load_agent(self, llm):
        agent = ReActAgent(
            tools = self.tools,
            llm = Settings.llm,
            verbose = True,
            streaming = False,
            system_prompt = "You are an AI assistant that helps users navigate and use the Learning Management System (LMS) effectively. Your job is to MUST choose the provided tool to answer the user’s question. Read the question and analyze it deeply to be able to choose the right tool to use."
        )
        return agent
    
if __name__ == "__main__":
    import asyncio
    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import ToolCallResult, AgentStream
    
    query = "Hướng dẫn tôi Tạo hoạt động Feedback"
    llm = Vllm(
        model = "meta-llama/Llama-3.1-8B",
        dtype = "float16",
        tensor_parallel_size = 2,
        temperature = 0,
        max_new_tokens = 4096,
        vllm_kwargs = {
            "swap_space": 1,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 4096,
        },
    )
    assistant = LMS_Assistant(llm)
    
    async def main():
        handle = assistant.agents.run(query, ctx= Context(assistant.agents))
        async for ev in handle.stream_events():
            if isinstance(ev, AgentStream):
                print(ev.response)
            elif isinstance(ev, ToolCallResult):
                print(ev.tool_output.content)
            else:
                print(ev)
                
    response = asyncio.run(main())
    print(str(response))
