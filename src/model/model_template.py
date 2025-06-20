from dataclasses import dataclass
from abc import ABC, abstractmethod

class BaseTemplate(ABC):

    @abstractmethod
    def __init__(self, model_name:str):
        self.model_name = model_name

    def apply_conversasion(self, question:str, answer:str ):
        pass


class APIModelTemplate(BaseTemplate):

    def __init__(self, model_name:str):
        super().__init__(model_name)

    def llama_4_conversation(self, question:str, answer:str):
        
        messages=  [{
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"Câu hỏi: {question}\nCâu trả lời: {answer}\n\nVui lòng tạo ra một chuỗi tư duy (chain of thought) từ câu hỏi và câu trả lời này. Hãy giải thích chi tiết từng bước trong quá trình tư duy của bạn."
            }
        ]
        }]
        return messages
    
    def apply_conversasion(self, question:str, answer:str):

        if 'meta-llama/llama-4' in self.model_name:
            return self.llama_4_conversation(question= question, answer= answer)
    
