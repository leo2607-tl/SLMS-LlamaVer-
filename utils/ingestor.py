import glob
import os
import sys
from pathlib import Path
import uuid
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from database.ingest import Ingest
from typing import List
from config.schema import Documents

class DataIngestion:
    def __init__(self, ingest_instance):
        self.ingest = ingest_instance
    
    def read_txt_files(self, folder_path: str) -> List[Documents]:
        documents = []

        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        if not txt_files:
            print(f"Không tìm thấy file .txt nào trong {folder_path}")
            return documents
        
        print(f"Tìm thấy {len(txt_files)} file .txt")
        
        for file_path in txt_files:
            try:
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()

                if not text:
                    print(f"Bỏ qua file rỗng: {file_path}")
                    continue
                
                file_name = os.path.basename(file_path)
                doc_id = str(uuid.uuid4()) 
                
                documents.append({
                    'doc_id': doc_id,
                    'text': text,
                    'metadata': {
                        'file_name': file_name,
                        'file_path': file_path
                    }
                })
                
                print(f"Đã đọc: {file_name} (Length: {len(text)} chars)")
                
            except Exception as e:
                print(f"Lỗi khi đọc file {file_path}: {str(e)}")
                continue
        
        return documents
    
    def ingest_from_folder(self, folder_path: str, batch_size: int = 100):
        documents = self.read_txt_files(folder_path)
        
        if not documents:
            return {
                'total_files': 0,
                'elasticsearch': 0,
                'qdrant': 0
            }
        
        total_results = {
            'total_files': len(documents),
            'elasticsearch': 0,
            'qdrant': 0
        }
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"\nIngest batch {i//batch_size + 1}: {len(batch)} documents")
            
            try:
                result = self.ingest._ingest(batch)
                total_results['elasticsearch'] += result['elasticsearch']
                total_results['qdrant'] += result['qdrant']
                print(f"Batch result: ES={result['elasticsearch']}, Qdrant={result['qdrant']}")
            except Exception as e:
                print(f"Lỗi khi ingest batch {i//batch_size + 1}: {str(e)}")
                continue
        
        return total_results


if __name__ == "__main__":
    ingest_instance = Ingest()
    
    data_ingestion = DataIngestion(ingest_instance)
    
    dataset_folder = "/workspace/aiclub02/leo/LMS_Assistant/dataset"
    
    print(f"Bắt đầu ingest từ folder: {dataset_folder}\n")
    results = data_ingestion.ingest_from_folder(
        folder_path=dataset_folder,
        batch_size=50  
    )
    
    print("\n" + "="*50)
    print("KẾT QUẢ INGEST:")
    print(f"Tổng số file: {results['total_files']}")
    print(f"Elasticsearch: {results['elasticsearch']} documents")
    print(f"Qdrant: {results['qdrant']} vectors")
    print("="*50)