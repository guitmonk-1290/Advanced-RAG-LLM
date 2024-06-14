from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from src.llama_query_pipeline import QueryExecutor
import uvicorn
import argparse

app = FastAPI()

_cache = {}

class InputText(BaseModel):
    inputText: str

class QueryParams(BaseModel):
    inputText: str
    choosen_tables: List[str]

def get_shared_instance():
    # get db_config for user
    # Later set up the db_config object dynamically
    db_config = {
        "host": "127.0.0.1",
        "user": "arinxd",
        "password": "eatdatass",
        "database": "spectra"
    }

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query Executor")
    parser.add_argument("--llm", type=str, default="openai", choices=["openai", "gemini", "ollama"],
                        help="Choose the LLM model service")
    parser.add_argument("--embeddings", type=str, default="openai", choices=["openai", "gemini", "ollama"],
                        help="Choose the Embeddings model service")
    args = parser.parse_args()

    if args.llm is None:
        args.llm = "openai"
    if args.embeddings is None:
        args.embeddings = "openai"

    print(args.embeddings)

    cache_key = tuple(db_config.items())  # Create key based on dictionary items
    if cache_key not in _cache:
        _cache[cache_key] = QueryExecutor(
            db_config=db_config,
            llm=args.llm,
            embedding=args.embeddings,
            ollama_embeddings_model="nomic-embed-text",
            batch_size=512
        )
    return _cache[cache_key]

@app.get("/")
async def base():
    return {"status": 200}


@app.post("/tables", response_model=dict, status_code=200)
async def tables(
    inputText: InputText,
    query_executor: QueryExecutor = Depends(get_shared_instance)
):
    try:
        related_tables = query_executor.get_related_tables(str(inputText))
        query_executor.set_tables(tables=related_tables)
        tnames = []
        for table in related_tables:
            tnames.append(table.table_name)
        return { "related_tables": tnames }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/response", response_model=dict, status_code=200)
async def response(
    params: QueryParams,
    query_executor: QueryExecutor = Depends(get_shared_instance)
):
    choosen_tables = params.choosen_tables
    inputText = params.inputText

    # Debug statements
    print("Received inputText:", inputText)
    print("Received choosen_tables:", choosen_tables)

    try:
        setted_tables = query_executor.get_setted_tables()
        schema_tables = [table for table in setted_tables if table.table_name in choosen_tables]
        query_executor.set_tables(schema_tables)
        SQL, nl_response = query_executor.run(query=inputText)

        return {"SQL": SQL, "response": nl_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# For direct inference without user-feedback
@app.post("/process-text", response_model=dict, status_code=200)
async def process_text(inputText: InputText):
    try:
        query_executor = QueryExecutor(db_config={
            "host": "127.0.0.1",
            "user": "<username>",
            "password": "<password>",
            "database": "<db_name>"
        })
        sql, nl_response, = query_executor.run(query=inputText.inputText)
        return {"SQLQuery": sql, "response": nl_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
