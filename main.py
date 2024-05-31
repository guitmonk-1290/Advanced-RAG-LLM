from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.llama_query_pipeline import QueryExecutor
import uvicorn

app = FastAPI()


class InputText(BaseModel):
    inputText: str


@app.get("/")
async def base():
    return {"status": 200}


@app.post("/process-text", response_model=dict, status_code=200)
async def process_text(inputText: InputText):
    try:
        query_executor = QueryExecutor(
            db_config={
                "host": "<host>",
                "user": "<username>",
                "password": "<password>",
                "database": "<db_name>",
            }
        )
        sql, response, sql_res = query_executor.run(query=inputText.inputText)
        return {"SQLQuery": sql, "response": response, "SQLResult": sql_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
