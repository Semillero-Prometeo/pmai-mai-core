from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class NatsSubscriber(BaseModel):
    controller: Callable[[dict[str, Any]], Any]
    subject: str
