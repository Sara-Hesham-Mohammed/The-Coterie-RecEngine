from pydantic import BaseModel
from typing import List

class User(BaseModel):
    user_id: int
    attendedEvents: List[int]

class RecommendationRequest(BaseModel):
    user: User
    all_events: List[int]
