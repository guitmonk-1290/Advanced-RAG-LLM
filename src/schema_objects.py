from llama_index.core.objects import (
    SQLTableSchema,
)

table_schema_objs = [
    (
        SQLTableSchema(
            table_name="ticket_managements",
            context_str="'ticket_managements' table stores information about ticket status, ticket incident date, due date, ticket title, ticket description, creation time, update time. Following are the status codes for a ticket: 1 means 'Open', 2 means 'in Progress', 3 means 'on hold', 4 means 'resolved' and 5 means 'closed'",
        )
    ),
    (
        SQLTableSchema(
            table_name="lead_managements",
            context_str="'lead_managements' table stores information about leads including lead name, lead id, lead description, lead status, lead refund status.",
        )
    ),
    (
        SQLTableSchema(
            table_name="license_issue_masters",
            context_str="'license_issue_masters' table stores information about licenses including license type, license id, production uat, license invoice, activate date, license status, license amc information.",
        )
    ),
    (
        SQLTableSchema(
            table_name="clients",
            context_str="'clients' table stores information about a client including client id, client name, client address, client code and status."
        )
    ),
]  # add a SQLTableSchema for each table
