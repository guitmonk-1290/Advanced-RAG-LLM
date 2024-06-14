from llama_index.embeddings.openai import OpenAIEmbedding

class openai:
    def __init__(
            self,
            model: str,
            batch_size: int
    ):
        self.embedding_model = OpenAIEmbedding(
            model=model,
            embed_batch_size=batch_size
        )
    
    def getModel(self):
        return self.embedding_model