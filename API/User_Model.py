from pydantic import BaseModel
from typing import List

class User(BaseModel):
    id: int
    age: int
    tags: List[str]
