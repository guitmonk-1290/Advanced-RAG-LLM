import datetime
import os
import re

os.environ["HF_HOME"] = "model/"

from pathlib import Path
from typing import Dict

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne

from .schema_objects import table_schema_objs
from typing import List
from llama_index.llms.ollama import Ollama
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.core.service_context import ServiceContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from sqlalchemy import create_engine, MetaData, text

from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from .utils import connect_to_DB, execute_query
import ollama


class QueryExecutor:
    def __init__(self, db_config):
        # Callbacks support token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.db_config = db_config

        # Connect to database
        self.engine = create_engine(
            f"mysql://{db_config['user']}:{db_config['password']}@localhost:3306/{db_config['database']}"
        )
        self.sql_database = SQLDatabase(self.engine, view_support=True)
        # print("[INFO] ENGINE DIALECT: ", self.engine.dialect.name)

        self.llm = Ollama(model="phi3", request_timeout=500000)

        # Initialize LLM model settings
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            base_url="http://127.0.0.1:11434",
            model_name="nomic-embed-text",
            embed_batch_size=512,
        )

        self.table_node_mapping = SQLTableNodeMapping(self.sql_database)

        if not Path("indexes").exists():
            # Get table node mappings

            self.obj_index = ObjectIndex.from_objects(
                table_schema_objs,
                self.table_node_mapping,
                VectorStoreIndex,
            )
            self.obj_index.persist(persist_dir="indexes")
        else:
            print(
                "[INFO] Found existing indexes. Loading from disk...\nNote: Delete the indexes directory to create new indexes\n"
            )
            # load index
            self.obj_index = ObjectIndex.from_persist_dir(
                "indexes", self.table_node_mapping
            )

        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=1)
        self.sql_connection = connect_to_DB(db_config)
        self.storage_context = None
        self.service_context = None
        self.vector_index_dict = None

        self.SQLQuery = ""

    def index_all_tables(
        self, table_index_dir: str = "table_index_dir"
    ) -> Dict[str, VectorStoreIndex]:
        """Index all tables."""
        if not Path(table_index_dir).exists():
            os.makedirs(table_index_dir)

        vector_index_dict = {}
        engine = self.sql_database.engine
        print(f"[LOG] Started indexing at {datetime.datetime.now()}")

        chroma_client = chromadb.PersistentClient()

        for table_name in self.sql_database.get_usable_table_names():
            # print(f"[LOG] Indexing rows in table: {table_name}")

            # if not os.path.exists(f"{table_index_dir}/{table_name}"):
            if f"{table_name}" not in [
                c.name for c in chroma_client.list_collections()
            ]:
                chroma_collection = chroma_client.create_collection(
                    name=f"{table_name}"
                )
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

                # get all rows from table
                # print(f"[INFO] Cannot find index directory for table [{table_name}]")
                with engine.connect() as conn:
                    cursor = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3;"))
                    result = cursor.fetchall()
                    row_tups = []
                    for row in result:
                        row_tups.append(tuple(row))

                # index each row, put into vector store index
                nodes = [TextNode(text=str(t)) for t in row_tups]

                self.storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

                # put into vector store index (use OpenAIEmbeddings by default)
                # print(f"[INFO] Writing indexes to chromaDB for {table_name}")
                index = VectorStoreIndex(
                    nodes,
                    service_context=self.service_context,
                    storage_context=self.storage_context,
                )
            else:
                # print(f"[INFO] Found existing indexes in chromaDB..")

                chroma_collection = chroma_client.get_collection(name=f"{table_name}")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            vector_index_dict[table_name] = index

        return vector_index_dict

    def get_table_context_and_rows_str(
        self, table_schema_objects: List[SQLTableSchema]
    ):
        """Get table context string."""
        context_strs = ""
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        # table_info = f"These are the table schemas for 2 tables named {table_schema_objs[0].table_name} and {table_schema_objs[1].table_name}\n"
        for table_schema_obj in table_schema_objects:
            table = metadata.tables[table_schema_obj.table_name]
            columns = table.columns

            table_info = ""
            table_info += f"CREATE TABLE `{table_schema_obj.table_name}` ("
            for column in columns:
                # Format column information
                column_info = f"'{column.name}' {column.type}"
                # Add information about constraints (e.g., primary key, not null)
                # if column.primary_key:
                #     column_info += " PRIMARY KEY"
                # if not column.nullable:
                #     column_info += " NOT NULL"
                if column.comment:
                    column_info += f" '{column.comment}'"
                table_info += f"{column_info}, "
            table_info += ")\n"

            if table_schema_obj.context_str:
                # table_opt_context = " The table description is: "
                table_opt_context = table_schema_obj.context_str
                table_info += table_opt_context

            # context_strs.append(table_info)
            context_strs += f"{table_info}\n"
        # return "\n\n".join(context_strs)
        return context_strs

    def get_table_context_str(self, table_schema_objects: List[SQLTableSchema]):
        """Get table context string"""
        context_strs = []
        for table_schema_obj in table_schema_objects:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            print(f"[TABLE_INFO] {table_info}")

            context_strs.append(table_info)

        print(f"[CONTEXT]: {context_strs}")
        return "\n\n".join(context_strs)

    def parse_response_to_sql(self, response: str) -> str:
        """Parse response to SQL."""
        sql_pattern = r"(?i)(SELECT)\s+.*?;"

        # Search for the SQL query in the input string
        match = re.search(sql_pattern, response, re.DOTALL)

        if match:
            self.SQLQuery = match.group(0)
        else:
            self.SQLQuery = None

        return self.SQLQuery

    def parse_nl_response(self, response: str) -> str:
        """Parse NL response"""
        lines = response.split(".")
        # Get the first line (if it exists)
        first_line = lines[0] if lines else ""
        return first_line

    def run(self, query: str):
        print(f"EXECUTION STARTED AT {datetime.datetime.now()}")
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=Settings.embed_model
        )
        self.vector_index_dict = self.index_all_tables()

        table_schema_objects = self.obj_retriever.retrieve(query)
        for table in table_schema_objects:
            print(f"[matched_table]: {table.table_name}")

        context_str = self.get_table_context_and_rows_str(
            table_schema_objects=table_schema_objects
        )
        context_str += f"According to this table schema, write a syntactically correct SQL query as mentioned in the following query and do not make up any information. ONLY write the SQL query and nothing else. query: '{query}'"
        print(f"[LLM_CONTEXT_STR]: {context_str}")

        # Firstly, pull the model from ollama by 'ollama pull deepseek-coder'
        response = ollama.chat(
            model="deepseek-coder",
            messages=[{"role": "user", "content": context_str}],
            stream=False,
            keep_alive=True,
        )

        print(f"[RESPONSE: {response}")
        sql_parsed = self.parse_response_to_sql(response["message"]["content"])
        print(f"[LOG] Finished processing at {datetime.datetime.now()}")
        print(f"[PARSED_SQL]: {sql_parsed}")

        sql_res = execute_query(self.sql_connection, sql_parsed)
        print(f"[SQL_RES]: {sql_res}")

        response_synthesis_prompt_str_system = """
You are a chatbot that takes an input question about a database and responds to the input question in natural language. 
Given an input question and the SQL result for that question from a SQL database, combine the input and the SQL result into a concise single-line natural language response to answer the input question and DO NOT make up any information.
Follow these guidelines to generate your response:
- If the SQL response is an empty list, simply give the response that there is no such result found.
- ONLY return the single-line answer based on the SQL Answer

Here are some relevant examples:

Input Question: what is the status of ticket with id '4556'
SQL result: [{"status": "2"}]
AI response: The status of the ticket is '2'.

Input Question: what is the incident date of ticket with id '4556'
SQL result: [{"incident_date": "2024-03-05"}]
AI response: The incident date of the ticket is "2024-03-05"
"""

        response_synthesis_prompt_str_user = f"""
Answer this question based on the SQL answer and return only a very short single-line chatbot response directly answering the question and NOTHING ELSE.
Input Question: {query}
SQL result: {sql_res if len(sql_res) else 'There is no such result in the database'}
        """

        print(f"[prompt]: {response_synthesis_prompt_str_user}")

        # print(f"[NL_RESPONSE_PROMPT]: {response_synthesis_prompt_str}")

        response = ollama.chat(
            model="tinyllama:1.1b-chat",
            messages=[
                {"role": "system", "content": response_synthesis_prompt_str_system},
                {"role": "user", "content": response_synthesis_prompt_str_user},
            ],
            stream=False,
            keep_alive=True,
        )

        # response_parsed = self.parse_nl_response(response["message"]["content"])
        print(f"[NL_RESPONSE]: {response['message']['content']}")

        print(f"EXECUTION FINISHED AT {datetime.datetime.now()}")
        return sql_parsed, response["message"]["content"], sql_res