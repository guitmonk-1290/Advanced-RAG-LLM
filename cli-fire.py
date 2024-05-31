#!/usr/bin/env python

import fire
from src.llama_query_pipeline import QueryExecutor

if __name__ == "__main__":
    fire.Fire(QueryExecutor)