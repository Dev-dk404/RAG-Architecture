from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

DATA_PATH = 'data/'

# Loads the data and returns document list
def load_data():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Further split documents regulating the chunk size and include overlap for providing context   
def split_documents(documents: list[Document]):
    text_splitter =  RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 250
    )
    return text_splitter.split_documents(documents)


# Calculating chunk_id in the format source:page:page_repetition and adding to metadata of chunk
def calculate_chunk_id(chunks: list[Document]):
    previous_page_id = None
    chunk_index = 0

    for chunk in chunks:
        print()
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if (current_page_id == previous_page_id):
            chunk_index += 1
        else: chunk_index = 0

        chunk_id = f"{current_page_id}:{chunk_index}"
        previous_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

        
        


        