from pymongo import MongoClient, UpdateOne, DeleteMany, errors
import threading
import socket
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure


class DatabaseConnection:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseConnection, cls).__new__(cls)
                # Update the connection string to include the database name
                cls._instance.client = MongoClient('mongodb://init_user:init_pwd@mongodb:27017/tdr_case', serverSelectionTimeoutMS=5000)
                # cls._instance.client = MongoClient('mongodb://init_user:init_pwd@127.0.0.1:27017/tdr_case', serverSelectionTimeoutMS=5000)
        return cls._instance

    def get_db(self):
        return self._instance.client.get_default_database()


# Replace global variable and initialize_database function with DatabaseConnection usage
db_conn = DatabaseConnection().get_db()


def log_error_to_database(error_message):
    """
    Records error details into the database and prints them to the console.

    Parameters:
    - db: The database connection object.
    - error_message (str): The error message to be logged.
    - retry_count (int): The current retry count for the task.
    """
    print(f"Error logged to database: {error_message}")
    db_conn.error_logs.insert_one({'error_message': error_message})


def count_tasks_by_status():
    """
    Counts the number of tasks in the 'tasks' collection of the database, grouped by their status.

    :return: A dictionary with the count of tasks for each status.

    Args:
        local_db:
    """
    # Aggregation pipeline for grouping and counting documents by status
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]

    # Execute the aggregation pipeline
    results = db_conn['tasks'].aggregate(pipeline)

    # Convert the aggregation results into a readable dictionary
    status_counts = {doc['_id']: doc['count'] for doc in results}

    return status_counts


def find_one(collection_name, query):
    """
    Finds and returns a single document from the specified collection based on the provided query.

    :param collection_name: The name of the collection to query.
    :param query: The query criteria to find the document.
    :return: A single document (dict) or None if no document matches.
    """
    collection = db_conn[collection_name]
    return collection.find_one(query)


def find_all(collection_name, query):
    """
    Finds and returns all documents from the specified collection based on the provided query.

    :param collection_name: The name of the collection to query.
    :param query: The query criteria to find documents.
    :return: A list of documents (dicts) matching the query or an empty list if no documents match.
    """
    collection = db_conn[collection_name]
    return list(collection.find(query))


def update_database_task_status(collection_name, task_id, new_status):
    """
    Updates the status of a task in the database.
    :param collection_name: The name of the collection to perform the bulk updates.
    :param task_id: The unique identifier of the task.
    :param new_status: The new status to set for the task (e.g., 'completed', 'failed').
    :return: None
    """
    collection = db_conn[collection_name]

    result = collection.update_one({'_id': task_id}, {'$set': {'status': new_status}})

    if result.matched_count > 0:
        print(f"Task with ID {task_id} updated to status '{new_status}'.")
    else:
        print(f"No task found with ID {task_id} to update.")


def bulk_update(collection_name, updates, upsert=False):
    """
    Performs bulk update operations on the specified collection. This function allows for multiple
    update operations to be sent to the database in a single command, improving performance
    for large batches of updates.

    Parameters:
    - collection_name (str): The name of the collection to perform the bulk updates.
    - updates (list of dict): A list of update operations to be performed. Each dictionary in the list
      should have 'filter' and 'update' keys. 'filter' is the criteria for selecting documents to update,
      and 'update' is the update operation to apply to the selected documents.
    - upsert (bool): If True, each update operation will create a new document if no existing document
      matches the filter. Defaults to False.

    Returns:
    - A pymongo.bulk.BulkWriteResult object which contains information about the bulk operation's outcome.
      This includes the count of matched, modified, inserted, and upserted documents.

    Example of an update operation:
    updates = [
        {
            'filter': {'key1': 'value1'},
            'update': {'$set': {'key2': 'new_value'}}
        },
        # ... more update operations ...]

    Usage:
    result = bulk_update('my_collection', updates, upsert=True)
    """
    collection = db_conn[collection_name]
    operations = [UpdateOne(update['filter'], update['update'], upsert=upsert) for update in updates]
    result = collection.bulk_write(operations)
    return result


def insert_many(collection_name, documents):
    """
    Inserts multiple documents into the specified collection.
    Continues insertion even if some documents fail (e.g., due to duplication).

    Parameters:
    - collection_name (str): Name of the collection.
    - documents (list of dict): Documents to be inserted.

    Returns:
    - None. Errors and successes are logged.
    """
    collection = db_conn[collection_name]
    documents = [doc.copy() for doc in documents]

    try:
        collection.insert_many(documents, ordered=False)
        print(f"Inserted documents into {collection_name}.")
    except errors.BulkWriteError as e:
        write_errors = e.details.get('writeErrors', [])
        for error in write_errors:
            if error['code'] != 11000:  # Ignore duplicate key errors
                print(f"Error: {error}")
            else:
                print(f"Ignoring duplicate key error for document: {error['op']}")


def upsert_many(collection_name, documents, unique_field):
    """
    Upserts multiple documents into the specified collection based on a unique field.

    Parameters:
    - collection_name (str): Name of the collection.
    - documents (list of dict): Documents to be upserted.
    - unique_field (str): Field name to determine uniqueness.

    Returns:
    - None. Errors and successes are logged.
    """
    collection = db_conn[collection_name]

    for doc in documents:
        query = {unique_field: doc.get(unique_field)}
        update = {"$set": doc}
        try:
            result = collection.update_one(query, update, upsert=True)
            if result.upserted_id is not None:
                print(f"Inserted new document with {unique_field} = {doc.get(unique_field)}")
            else:
                print(f"Updated existing document with {unique_field} = {doc.get(unique_field)}")
        except Exception as e:
            print(f"Error upserting. Document not added. Exception: {e}")

    # Usage example
    # upsert_many('tasks', new_tasks, 'params.url')


def upsert_one(collection_name, query, data):
    """
    Inserts or updates a single document in the specified collection.

    :param collection_name: The name of the collection.
    :param query: Query to match the document.
    :param data: The document to be inserted or updated.
    :return: Result of the operation.
    """
    collection = db_conn[collection_name]
    result = collection.update_one(query, {'$set': data}, upsert=True)

    if result.upserted_id:
        print(f"Inserted a new document with data: {data}")
        return result.upserted_id
    else:
        print(f"Updated existing document(s) with data: {data}")
        return result.matched_count


def insert_one(collection_name, data):
    """
    Inserts a single document into the specified collection.

    :param collection_name: The name of the collection where the document will be inserted.
    :param data: The document to be inserted.
    :return: The result of the insert operation (e.g., ID of the inserted document).
    """
    collection = db_conn[collection_name]
    result = collection.insert_one(data)
    print(f"Inserted a new document with data: {data}")
    return result.inserted_id


def update_one(collection_name, filter_query, update_data):
    """
    Updates a single document in the specified collection based on the provided filter and update data.

    :param collection_name: The name of the collection to query.
    :param filter_query: The query criteria to find the document.
    :param update_data: The update operation to be applied to the document.
    :return: Result of the update operation.
    """
    collection = db_conn[collection_name]
    result = collection.update_one(filter_query, update_data)
    if result.matched_count > 0:
        print(f"Updated document in {collection_name}.")
    else:
        print(f"No document found with the given criteria in {collection_name}.")
    return result


def bulk_delete(collection_name, filters):
    """
    Performs bulk delete operations on the specified collection. This function allows for multiple
    delete operations to be sent to the database in a single command, improving performance
    for large batches of deletions.

    Parameters:
    - collection_name (str): The name of the collection to perform the bulk deletes.
    - filters (list of dict): A list of criteria for selecting documents to delete. Each dictionary in the list
      should represent a filter to select documents for deletion.

    Returns:
    - A pymongo.bulk.BulkWriteResult object which contains information about the bulk operation's outcome,
      including the count of deleted documents.

    Example of a delete operation:
    filters = [
        {'key1': 'value1'},
        # ... more filter criteria ...]

    Usage:
    result = bulk_delete('my_collection', filters)
    """
    collection = db_conn[collection_name]
    operations = [DeleteMany(line_filter) for line_filter in filters]
    result = collection.bulk_write(operations)
    return result


def delete_one(collection_name, db_filter):
    """
    Deletes a single document from the specified collection based on the provided filter.

    :param collection_name: The name of the collection where the document will be deleted.
    :param db_filter: The criteria for selecting the document to delete.
    :return: The result of the delete operation, typically the count of deleted documents.
    """
    collection = db_conn[collection_name]
    result = collection.delete_one(db_filter)
    print(f"Deleted document with filter: {db_filter}")
    return result.deleted_count


def close_connection(client):
    """
    Closes the database connection.
    :param client: The MongoDB client instance to close.
    :return: None
    """
    client.close()


def debug_database_connection(hostname, port, username, password, database_name, collection_name):
    try:
        # Diagnose Hostname Resolution
        print("Testing whether hostname is resolvable...")
        socket.gethostbyname(hostname)
        print("Hostname is resolvable")

        # Establishing a MongoDB connection
        print("Testing port accessibility and authentication...")
        client = MongoClient(f'mongodb://{username}:{password}@{hostname}:{port}/{database_name}?authSource=admin',
                             serverSelectionTimeoutMS=5000)
        client.server_info()  # This will test port accessibility and authentication
        print("Port is accessible and authentication is successful")

        # Diagnose Database Accessibility
        print("Testing database accessibility...")
        db = client[database_name]
        db.command("ping")
        print("Database is accessible")

        # Diagnose Collection Accessibility
        print("Testing collection accessibility...")
        collection = db[collection_name]
        collection.find_one()
        print("Collection is accessible")

    except socket.gaierror:
        print("Hostname resolution failed")

    except ServerSelectionTimeoutError as timeout:
        print(f"Port is not accessible or authentication failed: {timeout}")

    except OperationFailure as opfailure:
        print(f"Authentication failed or database/collection is not accessible: {opfailure}")
