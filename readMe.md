# LLAMA-Based Retrieval-Augmented Generation (RAG) for PDF Data

I am building a RAG system, which goes through local pdf data, card game rule books in this project, and give its answers based of it. I am using uno, poker and marriage card game rule books as source of data for this RAG system. I am using a LLaMA model.

## Methodologies

### Data Loader
In this phase, we extract the text data from each page of the pdfs and organize it into content with metadata. It iterates through pdfs, extract text which becomes content and creates title based on pdf title and page number. It becomes metadata which is useful for indexing the contents of pdf at page level.
Code for this is present in [data_loader.py](data_loader.py)
The data in each page might be too huge, so we split the data, reffered as Document here, into chunks with appropriate size using split_documents() function in [data_loader.py](data_loader.py). 

### Data Embeddings
In order to faster retrieval of data, the chunks will be embedded into numerical form in vector space