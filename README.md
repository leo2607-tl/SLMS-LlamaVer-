# SIU_LMS_Assistant

`SIU_LMS_Assistant` là một hệ thống trợ lý AI cho Learning Management System (LMS), sử dụng Retrieval-Augmented Generation (RAG) với Elasticsearch và Qdrant để trả lời câu hỏi dựa trên tài liệu. Dự án sử dụng FastAPI để cung cấp API, `llama_index` và `vllm` cho mô hình ngôn ngữ, cùng một agent logic để điều phối truy vấn.

## Tổng quan

- FastAPI backend cho truy vấn AI
- Hệ thống RAG kết hợp Elasticsearch + Qdrant
- Embedding với HuggingFace và LlamaIndex
- Agent ReAct sử dụng các công cụ truy vấn ngữ cảnh
- Hỗ trợ ingest tài liệu `.txt` từ thư mục `dataset`
- Docker Compose nâng cao cho Elasticsearch và Qdrant

## Kiến trúc chính

- `services/api.py` - entry point FastAPI, khởi tạo LLM và agent, cung cấp endpoint `/query`
- `agent/assistant.py` - định nghĩa `LMS_Assistant`, load tool và agent workflow
- `config/config.py` - cấu hình hệ thống, endpoint Elasticsearch và Qdrant, model embedding/generation
- `database/` - kết nối DB và logic ingest
  - `elasticsearch_database.py` - kết nối và quản lý index Elasticsearch
  - `qdrant_database.py` - kết nối và quản lý collection Qdrant
  - `ingest.py` - ingest tài liệu vào Elasticsearch và Qdrant
- `utils/` - công cụ bổ trợ
  - `ingestor.py` - đọc file `.txt` từ thư mục dataset và ingest
  - `elasticsearch_utils.py`, `qdrant_utils.py` - debug / inspect dữ liệu
- `docker/docker-compose.yml` - cấu hình Elasticsearch và Qdrant service

## Yêu cầu

- Python 3.11+ (hoặc 3.10+)
- GPU nếu dùng `vllm` và mô hình lớn
- Docker / Docker Compose (để chạy Elasticsearch + Qdrant)
- Token Hugging Face nếu dùng model private hoặc cần tải model từ HF

## Cài đặt

1. Mở thư mục dự án:
   ```bash
   cd /home/leo/workspace/SIU_LMS_Assistant
   ```
2. Tạo môi trường ảo và kích hoạt:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Cài dependencies cần thiết (chỉnh theo môi trường nếu cần):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install fastapi uvicorn python-dotenv llama-index transformers vllm qdrant-client elasticsearch numpy
   ```

## Cấu hình môi trường

Tạo file `.env` trong thư mục gốc với các biến sau:

```bash
HUGGING_FACE_TOKEN=<your_huggingface_token>
```

Nếu bạn muốn sửa đường dẫn cache hoặc model, cập nhật trực tiếp trong `config/config.py`.

## Khởi động dịch vụ

1. Khởi chạy Elasticsearch và Qdrant:
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```
2. Chạy FastAPI app:
   ```bash
   python services/api.py
   ```
3. API sẽ lắng nghe mặc định tại:
   - `http://0.0.0.0:8000`
4. Gọi endpoint truy vấn:
   - `POST http://localhost:8000/query`
   - Body JSON: `{ "query": "Câu hỏi của bạn" }`

## Ingest dữ liệu

- Các tài liệu `.txt` được đặt trong thư mục `dataset`
- Chạy script ingest:
  ```bash
  python utils/ingestor.py
  ```
- Script sẽ đọc các file, tạo embedding và lưu vào Elasticsearch + Qdrant.

## Ghi chú

- `config/config.py` định nghĩa các endpoint:
  - Elasticsearch: `http://localhost:9200`
  - Qdrant: `http://localhost:6333`
  - Chỉ mục / collection mặc định: `leo_lms_assistant`
- `agent/assistant.py` hiện sử dụng model `meta-llama/Llama-3.1-8B` cho testing và `meta-llama/Llama-3.2-3B` trong FastAPI.
- Nếu bạn không có GPU, hãy điều chỉnh model hoặc tham số `vllm` cho phù hợp.

## Mở rộng

- Thêm UI frontend cho người dùng
- Xây dựng endpoint authentication
- Bổ sung pipeline ingest hỗ trợ nhiều định dạng tài liệu hơn
- Thêm dashboard giám sát Elasticsearch / Qdrant

## Liên hệ

- Dự án này là trợ lý AI nội bộ cho LMS, thiết kế để trả lời câu hỏi dựa trên dữ liệu đã được ingest.
