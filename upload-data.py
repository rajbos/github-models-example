import os, time
from azure.storage.blob import BlobServiceClient

# Check if the AZURE_STORAGE_CONNECTION_STRING is set
if not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set")

def get_blogging_directory():
    # Create a new directory to clone the repository
    blogging_directory = "blog"
    if not os.path.exists(blogging_directory):
        print("Cloning the blogging repository")
        os.makedirs(blogging_directory)
        # Clone the blogging repository from github.com/rajbos/rajbos.github.io
        os.system(
            f"git clone https://github.com/rajbos/rajbos.github.io.git {blogging_directory}"
        )
    else:
        # check if the directory is more than 24 hours old
        if os.path.getmtime(blogging_directory) < time.time() - 60 * 60 * 24:
            print("Pulling the latest changes from the blogging repository")
            os.system(f"cd {blogging_directory} && git pull")

    # show the size of the files in the persist_dir
    blogging_directory = f"{blogging_directory}/_posts/"
    print("Size of the blogging directory:")
    os.system(f"du -shc {blogging_directory}")
    return blogging_directory

# Main code
blogging_directory = get_blogging_directory()
# push the blogging directory into an Azure Blob Storage
connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = "blogposts"
container_client = blob_service_client.get_container_client(container_name)

# Iterate over files in the blogging directory and upload each one
for root, dirs, files in os.walk(blogging_directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        blob_client = container_client.get_blob_client(file_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data)
        print(f"Uploaded {file_name} to the Azure Blob Storage")

print("Uploaded all blogposts to the Azure Blob Storage")
