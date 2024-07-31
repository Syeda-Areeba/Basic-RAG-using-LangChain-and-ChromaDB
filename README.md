# Basic-RAG-using-LangChain-and-ChromaDB

This project demonstrates how to read, process, and chunk PDF documents, store them in a vector database, and implement a Retrieval-Augmented Generation (RAG) system for question answering using LangChain and Chroma DB.

## Overview

The system reads PDF documents from a specified directory or a single PDF file, splits them into smaller chunks, and embeds these chunks into a vector database using GPT4All embeddings. A retriever is then used to fetch relevant chunks based on user queries, and a language model generates detailed responses.

## Features

- Read and process PDF documents.
- Split documents into manageable chunks.
- Store chunks in a vector database with embeddings.
- Implement a RAG system for question answering.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Syeda-Areeba/Basic-RAG-using-LangChain-and-ChromaDB.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Reading and Processing Documents

The `read_docs` function reads PDF files from a directory or a single file.

```python
import os
from PyMuPDFLoader import PyMuPDFLoader

def read_docs(path):
    if os.path.isdir(path):
        DATA = []
        for file in os.listdir(path):
            if file.endswith('.pdf'):
                loader = PyMuPDFLoader(path + file)
                data = loader.load()
                DATA.extend(data)
        return DATA
    elif path.endswith('.pdf'):  # for reading single pdf
        loader = PyMuPDFLoader(path)
        data = loader.load()
        return data
```

### Splitting Documents into Chunks

The `make_chunks` function splits documents into smaller chunks for better processing.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Size of each chunk in characters
        chunk_overlap=20,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.\n")
    document = chunks[1000]
    print(document.page_content)
    print(document.metadata)
    return chunks
```

### Embedding Chunks and Storing in Vector Database

The `to_vector_db` function embeds the chunks and stores them in a Chroma vector database.

```python
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)

def to_vector_db(chunks, embed):
    vector_db = Chroma.from_documents(chunks, embedding=embed)
    return vector_db

vector_store = to_vector_db(all_chunks, embeddings)
```

### Question Answering with RAG

The QA chain retrieves relevant chunks and generates a response using a language model.

```python
from langchain_community.llms import GPT4All
from langchain.prompts import hub
from langchain.output_parsers import StrOutputParser
from langchain.pipelines import RunnablePassthrough

model = GPT4All(model="orca-mini-3b-gguf2-q4_0.gguf", max_tokens=2048)
rag_prompt = hub.pull('rlm/rag-prompt')

def format_docs(documents):
    return '\n\n'.join(d.page_content for d in documents)

retriever = vector_store.as_retriever()

qa_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt | model | StrOutputParser()
)

response = qa_chain.invoke('Explain TCP.')
print(response)
```

## How It Works

1. **Reading Documents**: The `read_docs` function reads PDF files from a directory or a single file.
2. **Making Chunks**: The `make_chunks` function splits documents into smaller chunks for better processing.
3. **Embedding and Storing**: The `to_vector_db` function embeds the chunks and stores them in a Chroma vector database.
4. **Question Answering**: The QA chain retrieves relevant chunks and generates a response using a language model.

## Requirements

- LangChain
- Chroma DB
- PyMuPDFLoader
- GPT4All Embeddings
- GPT4All Language Model

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- LangChain: [https://github.com/langchain/langchain](https://github.com/langchain/langchain)
- Chroma DB: [https://github.com/chroma-db/chroma](https://github.com/chroma-db/chroma)
- GPT4All: [https://github.com/nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all)
