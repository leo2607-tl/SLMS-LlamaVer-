import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()

from agent.assistant import LMS_Assistant
from llama_index.llms.vllm import Vllm
import uvicorn
import fastapi
import asyncio
from llama_index.core.workflow import Context

HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
os.environ["HF_HOME"] = "/workspace/aiclub02/leo/.cache/vLLM/models/"

app = fastapi.FastAPI()

llm = Vllm(
    model = "meta-llama/Llama-3.2-3B",
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

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.post("/query")
async def query(query: str):
    response = await assistant.agents.run(query, ctx = Context(assistant.agents))
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)