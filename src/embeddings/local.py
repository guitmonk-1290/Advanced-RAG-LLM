from llama_index.embeddings.ollama import OllamaEmbedding

class local_ollama:
    def __init__(
            self,
            model: str,
            batch_size: int
    ):
        self.embedding_model = OllamaEmbedding(
            base_url="http://127.0.0.1:11434",
            model_name=model,
            embed_batch_size=batch_size
        )
    
    def getModel(self):
        return self.embedding_model