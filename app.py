from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from flask import Flask
from dotenv import load_dotenv
import os


def upload_document_to_blob_storage(file_path, container_name, blob_name, connection_string):
    try:
        # Create BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Get a container client
        container_client = blob_service_client.get_container_client(container_name)

        # Create a blob client
        blob_client = container_client.get_blob_client(blob_name)

        # Upload the document to blob storage
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)

        print(f"Document '{blob_name}' uploaded to Azure Blob Storage successfully.")

    except Exception as e:
        print(f"Error uploading document to Azure Blob Storage: {e}")


load_dotenv()
app = Flask(__name__, template_folder="templates")

if __name__ == "__main__":
    print("Entered main")
    # Replace these values with your actual Azure Blob Storage information
    connection_string = os.getenv("CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME")
    blob_name = os.getenv("BLOB_NAME")
    file_path = os.getenv("FILE_PATH")


    upload_document_to_blob_storage(file_path, container_name, blob_name, connection_string)
    app.run(host='0.0.0.0', port = 8085)
    app.debug = True