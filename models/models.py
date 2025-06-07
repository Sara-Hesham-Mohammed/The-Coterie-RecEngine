from pydantic import BaseModel
from typing import List, Dict, Any

class Event(BaseModel):
    event_name: str
    location: str
    tags: set[str]
    is_paid: bool

class User(BaseModel):
    user_id: int
    location: str
    tags: set[str]
    history_events: List[Event]

class RecommendationRequest(BaseModel):
    user: User
    candidate_events: List[Event]

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
