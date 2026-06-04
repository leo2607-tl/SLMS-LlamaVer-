import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from database.elasticsearch_database import ElasticSearchDB 
from database.qdrant_database import QdrantDB

create_es = ElasticSearchDB(url = "http://localhost:9200")
create_qr = QdrantDB(url = "http://localhost:6333")

del_es = create_es.delete_collection()
del_qr = create_qr.delete_collection()

create_es.create_collection()
create_qr.create_collection(vector_size=1024, distance="Cosine")

es_count = create_es.counting()
qr_count = create_qr.counting()

print(f"Elasticsearch: {es_count}")
print(f"Qdrant: {qr_count}")