from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

llm = Ollama(
    model="llama3.1:8b",
    request_timeout=120.0,
    context_window=8000,
)

messages = [
    ChatMessage(
        role="system", content="You are an expert in the field of artificial intelligence"
    ),
    ChatMessage(role="user", content=""),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")