import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from typing import Dict
from typing import List
from elasticsearch import Elasticsearch
from config.config import DatabaseConfig
from config.schema import Response

class ElasticSearchDB:
    def __init__(
        self,
        url: str,
    ):
        
        self.dbcf = DatabaseConfig()
        self.index = self.dbcf.ELASTIC_SEARCH_INDEX
        self.es_client = Elasticsearch(url)
        
    def create_collection(self):
        
        index_setting = {
            'settings': {
                'analysis': {
                    'analyzer':
                        {
                            'default': {'type': 'english'}
                        }
                },
                'similarity': {
                    'default': {'type': 'BM25'}
                }
            },
            'mappings': {
                'properties': {
                    'doc_id': {'type': 'keyword'},
                    'text': {'type': 'text', 'analyzer': 'english'},                
                }
            }
        }

        if not self.es_client.indices.exists(index=self.index):
            self.es_client.indices.create(index=self.index, body=index_setting)
            print(f"Index {self.index} created")
        else:
            print(f"Index {self.index} already exists.")
                
    def delete_collection(self):
        if self.es_client.indices.exists(index=self.index):
            self.es_client.indices.delete(index=self.index)
            print(f"Index {self.index} deleted")
        else:
            print(f"Index {self.index} does not exist.")
            
    def get_all_nodes(self):
        query = {
            "query": {
                "match_all": {}
            }
        }
        response = self.es_client.search(index=self.index, body=query, size=1000)
        hits = response['hits']['hits']
        return hits
    
    def counting(self):
        query = {
            "query": {
                "match_all": {}
            }
        }
        count = self.es_client.count(index=self.index, body=query)['count']
        return count
    
    def search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict]:
        
        search_body = {
            "query": {
                "match": {
                    "text": query,
                }
            },
            "size": top_k
        }
        
        response = self.es_client.search(
            index=self.index,
            body=search_body
        )
        
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'doc_id': hit['_id'],
                'text': hit['_source']['text'],
                'score': hit['_score']
            })
        
        return results

