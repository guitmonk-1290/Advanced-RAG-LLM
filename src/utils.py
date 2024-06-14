import mysql.connector as sql
import datetime


def connect_to_DB(db_config):
    """Connects to a SQL database using provided configuration.

    Args:
        db_config (dict): Dictionary containing database connection details,
                         including host, user, password, database name.

    Returns:
        connection object: A connection object to the SQL database.

    Raises:
        Exception: If an error occurs while connecting.
    """
    try:
        connection = sql.connect(**db_config)
        return connection
    except sql.OperationalError as e:
        raise sql.OperationalError(f"Error connecting to DB: {e}") from e


def execute_query(connection, query):
    """Executes a SQL query and returns the results.

    Args:
        connection (object): A connection object to the SQL database.
        query (str): The SQL query string to execute.

    Returns:
        list or tuple: A list or tuple of rows containing the query results.

    Raises:
        Exception: If an error occurs while executing the query.
    """
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"[RESULTS]: {results}")

        # Get column names from cursor description
        columns = [col[0] for col in cursor.description]

        # Format the results
        formatted_results = [
            {
                columns[i]: (
                    value.strftime("%Y-%m-%d")
                    if isinstance(value, (datetime.datetime, datetime.date))
                    else value
                )
                for i, value in enumerate(row.values())
            }
            for row in results
        ]

        cursor.close()
        return formatted_results
    except sql.ProgrammingError as e:
        raise sql.ProgrammingError(f"Error executing query: {e}") from e

def get_fkey_info(connection, db_name, tables):
    """Get foreign keys for schema tables
    Args:
        connection (object): A connection object to the SQL database.
        db_name: The name of the database having the tables
        tables (str[]): schema tables

    Returns:
        list or tuple: A list or tuple of rows containing info about foerign keys of tables

    Raises:
        Exception: If an error occurs while executing the query.
    """
    try:
        query = f"""
SELECT 
    kcu.TABLE_NAME AS table_name,
    kcu.COLUMN_NAME AS foreign_key_column,
    kcu.REFERENCED_TABLE_NAME AS referenced_table,
    kcu.REFERENCED_COLUMN_NAME AS referenced_column
FROM 
    information_schema.KEY_COLUMN_USAGE AS kcu
JOIN 
    information_schema.TABLE_CONSTRAINTS AS tc 
    ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME 
    AND kcu.TABLE_NAME = tc.TABLE_NAME
WHERE 
    tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
    AND tc.CONSTRAINT_NAME = 'sk_fkey'
    AND kcu.TABLE_SCHEMA = '{db_name}'
    AND kcu.TABLE_NAME IN (%s);
        """

        tables = [table.table_name for table in tables]
        placeholder = ', '.join(f"'{_}'" for _ in tables)
        query = query % placeholder

        print("Query: ", query)

        cursor = connection.cursor()
        cursor.execute(query)

        # Fetch all the rows
        foreign_keys = cursor.fetchall()
        fkey_info = []

        # Print each foreign key information in the specified format
        for fk in foreign_keys:
            fkey_info.append(f"\n-- {fk[0]}.{fk[1]} can be joined with {fk[2]}.{fk[3]}")

        # Closing the cursor and connection
        cursor.close()
        return fkey_info
    except sql.ProgrammingError as e:
        raise sql.ProgrammingError(f"Error executing query: {e}") from e