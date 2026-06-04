from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

class ElasticSearchViewer:
    def __init__(self, es_client: Elasticsearch):
        self.client = es_client
    
    def show_all_documents(
        self, 
        index_name: str,
        size: int = 1000,
        query: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Lấy toàn bộ documents từ Elasticsearch index
        
        Args:
            index_name: Tên index
            size: Số lượng documents mỗi lần truy vấn (default: 1000)
            query: Truy vấn tùy chọn (default: None)
            
        Returns:
            List các documents với đầy đủ thông tin
        """
        all_documents = []
        body = {
            "query": query if query else {"match_all": {}},
            "size": size,
            "from": 0
        }
        
        while True:
            response = self.client.search(
                index=index_name,
                body=body
            )
            
            hits = response['hits']['hits']
            all_documents.extend(hits)
            
            if len(hits) < size:
                break
                
            body['from'] += size
        
        return all_documents
    
    def display_documents(
        self, 
        index_name: str,
        max_display: Optional[int] = None
    ):
        """
        Hiển thị documents từ Elasticsearch index
        
        Args:
            index_name: Tên index
            max_display: Số lượng documents tối đa để hiển thị (default: None - hiển thị tất cả)
        """
        documents = self.show_all_documents(index_name)
        
        if max_display:
            documents = documents[:max_display]
        
        for doc in documents:
            print(f"ID: {doc['_id']}")
            print(f"Source: {doc['_source']}")
            print("-" * 40)
        print(f"Tổng số documents hiển thị: {len(documents)}")
        
if __name__ == "__main__":
    from database.elasticsearch_database import ElasticSearchDB
    from config.config import DatabaseConfig
    es_db = ElasticSearchDB(url=DatabaseConfig.ELASTIC_SEARCH_URL)
    viewer = ElasticSearchViewer(es_db.es_client)
    
    index_name = DatabaseConfig.ELASTIC_SEARCH_INDEX
    viewer.display_documents(
        index_name=index_name,
        max_display=10
    )