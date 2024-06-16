![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
[![Build](https://github.com/guitmonk-1290/Advanced-RAG-LLM/actions/workflows/devops.yml/badge.svg)](https://github.com/guitmonk-1290/Advanced-RAG-LLM/actions/workflows/devops.yml)

# Advanced-RAG-LLM

This is the Back-end repo for the LLM API in the Advanced-RAG project encorporating feedback from the user

## Installation
```
pip install -r requirements.txt
```

## Run the server
```
python main.py
```

## Embeddings with vector store
We convert our database schema into vector embeddings to match user queries with the respective tables.
We have also incorporated a feedback-loop where user can give feedback on the selected tables and correct them.

## API Endpoints
1. To get the related tables to a user query
```
curl -X POST \
  http://localhost:5000/tables \
  -H "Content-Type: application/json" \
  -d '{"inputText": "how many clients are there?"}'
```

2. To get the response based on the choosen tables by the user
```
curl -X POST \
  http://localhost:5000/response \
  -H "Content-Type: application/json" \
  -d '{"inputText": "how many clients are there?", "choosen_tables": ["table1", "table2"]}'
```
The choosing of the tables is handled by the front-end logic

## Change Models
By default, it will use OpenAI for both embeddings and inference. You can change the ``models_config`` object to change the models for embeddings and inference accordingly

```
models_config = {
  "llm": "gemini",
  "embeddings": "ollama"
}
```

This code creates the APIs to interact with our choice of LLM service. These APIs then interacts with NodeJS which acts as an interface between oue front-end and the LLM APIs.<br>
<ul>
<li>Rest of the NodeJS code is available here: <a href="https://github.com/guitmonk-1290/Advanced-RAG-Node">Advanced-RAG-Node</a></li><br>
<li>The front-end angular code is available here: <a href="https://github.com/guitmonk-1290/Advanced-RAG-angular">Advanced-RAG-angular</a></li>
</ul>
