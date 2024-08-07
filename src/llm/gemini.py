from llama_index.llms.gemini import Gemini
import os
from dotenv import load_dotenv

load_dotenv()

class gemini:
    def __init__(
            self,
            model: str,
            **kwargs
    ):
        if os.getenv('GOOGLE_API_KEY') is None:
            raise ValueError("API key not found! Make sure that you have set the 'GOOGLE_API_KEY' variable in your environment")
        else:
            self.llm = Gemini(model=model)
    
    def infer(
            self,
            prompt: str
    ):
        response = self.llm.complete(prompt)
        return response
    
    def changeModel(
            self,
            model: str
    ):
        self.llm = Gemini(model=model)
    
    def getLLM(self):
        return self.llm