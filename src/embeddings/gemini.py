from llama_index.embeddings.gemini import GeminiEmbedding

class gemini:
    def __init__(
            self,
            model: str,
            batch_size: int
    ):
        self.embedding_model = GeminiEmbedding(
            model_name=model,
            embed_batch_size=batch_size
        )
    
    def getModel(self):
        return self.embedding_model