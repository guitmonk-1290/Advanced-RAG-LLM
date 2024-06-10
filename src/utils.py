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
