from pydantic import BaseModel


class Health(BaseModel):
    status: str
    timestamp: str
    msName: str
