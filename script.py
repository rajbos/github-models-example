import sys, requests, time, os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import load_index_from_storage
from utils import setup, get_blogging_directory, get_index, log_duration, parse_blog_header_date, convert_filename_to_url, get_github_rate_limit

# Main script
if __name__ == "__main__":

    # Set up the environment and initialize the models
    setup()

    # Check the current rate limit of using GitHub Models
    start_remaining_tokens, start_remaining_requests = get_github_rate_limit()

    # Get the blogging directory
    blogging_directory = get_blogging_directory()

    # Get the index
    index = get_index(blogging_directory)

    startTime = time.time()
    
    # Get the retriever from the index
    retriever = index.as_retriever()

    # Example prompts
    prompt = "Explain GitHub tokens and how to use them"
    prompt = "How can you use GitHub Actions with security in mind?"

    # Retrieve the fragments that match the question
    fragments = retriever.retrieve(prompt)
    log_duration(startTime, "Retrieval")

    # Load the documents that contain the fragments
    documents = []
    for fragment in fragments:
        # find the document based on the node_id
        document = index.storage_context.docstore.get_document(fragment.node_id)
        if document:
            # find the file name from this document
            file_name = document.metadata.get("file_name")
            print(f"> Document with [{fragment.node_id}] was found in file [{file_name}]")
            #print(f"> Fragment content: {fragment.text}")
            documents.append(file_name)
        else:
            print(f"[ERROR] Document with {fragment.node_id} was not found")

    # deduplicate the documents
    documents = list(set(documents))
    print(f"Found [{len(documents)}] documents that match the question:")

    # Load the content of the documents
    documentsContent = []
    for document in documents:
        # check if the file exists to prevent errors
        if not os.path.exists(f"{blogging_directory}{document}"):
            print(f"- [ERROR] File [{blogging_directory}{document}] does not exist")
            continue

        # load the file from the documents directory
        with open(f"{blogging_directory}{document}", "r") as file:
            content = file.read()
            documentsContent.append(content)

        # load the information from the content to show a reference
        date = parse_blog_header_date(content)
        url = convert_filename_to_url(document, date, "https://devopsjournal.io/blog")
        print(f"- File: [{blogging_directory}{document}] which leads to [{url}]")

    # Join the fragments into a single context string
    context = "\n------\n".join([ fragment.text for fragment in fragments ])

    # Prepare the prompt messages with the context data
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that answers some questions with the help of some context data.\n\nHere is the context data:\n\n" + context),
        ChatMessage(role="user", content=prompt)
    ]

    startTime = time.time()
    print(f"Calling the model with the following prompt: [{prompt}] and the context in all fragments")
    response = Settings.llm.chat(messages)
    print()
    print(response)
    print()
    log_duration(startTime, "Model call")
    get_github_rate_limit()

    # run the same prompt with the file content
    startTime = time.time()
    print(f"Calling the model with the following prompt: [{prompt}] and the context in all file content")

    # Join the documentsContent list into a single string
    documents_content_str = "\n".join(documentsContent)
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant that answers some questions with the help of some context data.\n\nHere is the context data:\n\n" + documents_content_str),
        ChatMessage(role="user", content=prompt)
    ]
    response = Settings.llm.chat(messages)
    print()
    print(response)
    print()
    log_duration(startTime, "Model call")
    
    remaining_tokens, remaining_requests = get_github_rate_limit()
    # show the used tokens and requests
    print(f"Total tokens used: {start_remaining_tokens - remaining_tokens}")
    print(f"Total requests used: {start_remaining_requests - remaining_requests}")
