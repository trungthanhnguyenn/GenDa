from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, model_name_or_path:str):

        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def generate(self, inputs: Any, max_tokens: int) -> Any:
        """"
        Run model generate
        """
        pass