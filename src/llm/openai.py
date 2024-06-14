from llama_index.llms.openai import OpenAI
import os

class openai:
    def __init__(
            self,
            model: str,
            **kwargs
    ):
        if not os.environ['OPENAI_API_KEY']:
            raise ValueError("API key not found! Make sure that you have set the 'OPENAI_API_KEY' variable in your environment")
        else:
            self.llm = OpenAI(model=model)
    
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
        self.llm = OpenAI(model=model)

    def getLLM(self):
        return self.llm