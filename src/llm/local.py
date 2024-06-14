import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

class local_ollama:
    def __init__(
            self,
            model: str,
            **kwargs
    ):
        self.model = model
        self.llm = Ollama(model=self.model)
    
    def chat(
            self,
            keep_alive,
            messages: list[any]
    ):
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=False,
            keep_alive=keep_alive
        )

        return response
    
    def generate(
            self,
            keep_alive,
            prompt: str
    ):
        response = ollama.generate(
            model = self.model,
            prompt=prompt,
            keep_alive=keep_alive
        )

        return response
    
    def setModel(
            self,
            model: str
    ):
        self.model = model
    
    def getLLM(
            self,
    ):
        return self.llm