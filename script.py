import os, time, dotenv, logging, sys, requests
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

def setup():
    # Load the environment variables
    dotenv.load_dotenv()

    # Check if the GITHUB_TOKEN is set
    if not os.getenv("GITHUB_TOKEN"):
        raise ValueError("GITHUB_TOKEN is not set")

    # Set the OPENAI_API_KEY to the GITHUB_TOKEN
    os.environ["OPENAI_API_KEY"] = os.getenv("GITHUB_TOKEN")
    os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com/"

    # Set up the logging
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO
    )  # change the logging.DEBUG for more verbose output
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Set up the LLM and Embedding models
    llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

def get_github_rate_limit():
    # send a get request to the openai api to check if the api key is valid
    response = requests.post("https://models.inference.ai.azure.com/chat/completions", headers={
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }, json={
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions"
            },
            {
                "role": "user",
                "content": "What is the capital of the Netherlands?"
            }]
    })
    if response.status_code != 200:
        raise ValueError(f"OPENAI_API_KEY is not valid: {response.text}")
    else:
        # Extract headers into a dictionary
        headers_dict = {header: value for header, value in response.headers.items()}
        print(f"X-Ratelimit-Remaining-Tokens: {headers_dict.get('x-ratelimit-remaining-tokens')}")
        print(f"X-Ratelimit-Remaining-Requests: {headers_dict.get('x-ratelimit-remaining-requests')}")

def log_duration(start_time, message):
    duration = round((time.time() - start_time) * 1000)
    print(f"{message} took [{duration}] ms")

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
        # check if the directory is more then 24 hours old
        if os.path.getmtime(blogging_directory) < time.time() - 60 * 60 * 24:
            print("Pulling the latest changes from the blogging repository")
            os.system(f"cd {blogging_directory} && git pull")

    # show the size of the files in the persist_dir
    blogging_directory = f"{blogging_directory}/_posts/"
    print("Size of the blogging directory:")
    os.system(f"du -shc {blogging_directory}")
    return blogging_directory

def get_index(blogging_directory): 
    persist_dir="blog_index"
    if not os.path.exists(persist_dir):
        print("Loading the data from the blogposts and create the index")
        startTime = time.time()
        documents = SimpleDirectoryReader(f"{blogging_directory}").load_data()
        index = VectorStoreIndex.from_documents(documents) # decrease batch size if needed for ratelimiting, insert_batch_size=150)

        log_duration(startTime, "Indexing")
        print("Persisting the index to the folder")
        startTime = time.time()
        # Persist the index to the folder
        index.storage_context.persist()
        log_duration(startTime, "Persisting")
        # show the size of the files in the persist_dir
        print("Size of the persisted directory:")
        os.system(f"du -sh {persist_dir}/*")
    else:
        print("Rebuilding storage context from directory")
        startTime = time.time()
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir),
        )
        index = load_index_from_storage(storage_context)
        log_duration(startTime, "Rebuilding storage context")
        # show the size of the files in the persist_dir
        print("Size of the persist directory:")
        os.system(f"du -sh {persist_dir}/*")
    return index

def convert_filename_to_url(document, date, base_url):
    # convert the filename to the blog url
    # file_name example: blog/_posts//2022-10-09-Enabling-GitHub-Actions-on-Enterprise-Server.md
    # actual blog url example: https://devopsjournal.io/blog/2022/10/08/Enabling-GitHub-Actions-on-Enterprise-Server
    file_name = document.split("/")[-1]
    file_name = file_name.replace(".md", "")
    # get the three date fields in the file name
    parts = file_name.split("-")
    year, month, day = parts[:3]
    title = "-".join(parts[3:])
    # remove the date from the filename
    file_name = file_name.replace(f"{year}-{month}-{day}-", "")
    # insert the date from the file itself in yyyy/mm/dd format
    # get year from date
    year = date.split(": ")[1].split("-")[0]
    month = date.split(": ")[1].split("-")[1]
    day = date.split(": ")[1].split("-")[2]
    # insert the date in the file name
    file_name = f"{year}/{month}/{day}/{file_name}"    
    return f"{base_url}/{file_name}"

def parse_blog_header_date(content):
    # parse the header into fields:
    #  ---
    # layout: post
    # title: "Enabling GitHub Actions on Enterprise Server: Common gotcha's"
    # date: 2022-10-08
    # tags: [GitHub, Actions, Enterprise, Runner, Dependabot, GitHub Actions, GitHub Enterprise Server, GHES, Gotcha's]
    # ---
    header = content.split("---")[1]
    header = header.split("\n")
    header = [line.strip() for line in header if line.strip()]
    # find the date tag
    date = [line for line in header if line.startswith("date")][0]
    return date

setup()

blogging_directory = get_blogging_directory()
index = get_index(blogging_directory)

startTime = time.time()
prompt = "Explain GitHub tokens and how to use them"
prompt = "How can you use GitHub Actions with security in mind?"
retriever = index.as_retriever()
fragments = retriever.retrieve(prompt)
log_duration(startTime, "Retrieval")

# Fragments that match the question:
documents = []
for fragment in fragments:
    #print(fragment)
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

    date = parse_blog_header_date(content)
    url = convert_filename_to_url(document, date, "https://devopsjournal.io/blog")
    print(f"- File: [{blogging_directory}{document}] which leads to [{url}]")

context = "\n------\n".join([ fragment.text for fragment in fragments ])

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
get_github_rate_limit()