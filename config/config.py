class DatabaseConfig:
    ELASTIC_SEARCH_URL = "http://localhost:9200"
    ELASTIC_SEARCH_INDEX = "leo_lms_assistant"
    QDRANT_URL = "http://localhost:6333"
    QDRANT_COLLECTION = "leo_lms_assistant"
    HF_EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
    HF_TOKENIZER_MODEL = "AITeamVN/Vietnamese_Embedding"
    HF_GENERATION_MODEL = "meta-llama/Llama-3.2-3B"
    SEMANTIC_WEIGHT = .7
    KEYWORD_WEIGHT = .3
    TOP_K = 2
    EMBEDDING_DEVICE = "cuda" 
    CACHE_DIR = "/workspace/aiclub02/leo/.cache/huggingface"

