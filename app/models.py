from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UploadResponse(BaseModel):
    job_id: str
    doc_id: str
    filename: str
    status: str
    cached: bool  # True if tree already exists for this doc

class JobStatus(BaseModel):
    job_id: str
    doc_id: str
    status: str   # received | processing | complete | failed
    progress: int # 0-100
    error: Optional[str] = None
    created_at: datetime

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: Optional[int]
    has_tree: bool
    created_at: datetime

class QueryRequest(BaseModel):
    doc_id: str
    query: str
    mode: str = "hierarchical"  # "hierarchical" | "long_context" | "both"