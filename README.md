# github-models-example
Example repo to use GitHub Models with a RAG example on my blog posts as the data source.

## Getting started
To get started with this repositor, open it using a GitHub Codespace. It has all the tools you need to start working with the data and GitHub Models.

After opening the repository in a Codespace, you can run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the script
To run the script, you can use the following command:

```bash
python script.py
```

### Process
The script will execute the following steps:

#### Step 1: Download and index the data
1. Download the data from my blogging repository at https://github.com/rajbos/rajbos.github.io
1. The data size from the repository is printed
1. Load the data using the LLamaIndex SDK from a folder in the blogging repository, into a VectorStoreIndex
1. Use the `text-embedding-3-small` OpenAI model from the GitHub Models to create embeddings for the data
1. Persists the embeddings in the `blog_index` folder to retrieve for the next run (if needed)
1. After persisting, the size of the new folder is printed

#### Step 2: Query the vector store
1. Query the vector store with the user prompt, e.g. "Explain GitHub tokens and how to use them"
1. The script will return the document fragments from the vector store that are most similar to the query
1. The fragments and the user prompt are now send to the `gpt-4o-mini` model to generate a natural language response on the query, using the fragments found in the vector store
1. The model response is printed

### Extra information
All along the way, the most interesting durationss for each step is shown to give you an idea of the performance of the script.
At the end of the script, the used API requests to GitHub Models are printed, together with the information about the used tokens, as this is all dependent on the [rate limit for GitHub Models](https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits).
