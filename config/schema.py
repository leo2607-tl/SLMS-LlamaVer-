from pydantic import BaseModel
 
class ElasticSearchResponse(BaseModel):
    doc_id: str
    text: str
    
class QdrantResponse(BaseModel):
    doc_id: str
    text: str

class Documents(BaseModel):
    doc_id: str
    text: str
    
class Response(BaseModel):
    doc_id: str
    text: str
    score: float
