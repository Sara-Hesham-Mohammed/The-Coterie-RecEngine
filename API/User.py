from pydantic import BaseModel


class User(BaseModel):
    id: int
    age: int
    tags: set[str]
