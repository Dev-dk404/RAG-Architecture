from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import os

def document_loader():
    data_path = 'data/'
    file_names = os.listdir(data_path)
    for file in file_names:
        file_path = data_path+ str(file)

        loader = PyPDFLoader(file_path)
        return loader.load_and_split()
    
def split_documents(documents: list[Document]):
    text_splitter =  RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100
    )
    return text_splitter.split_documents(documents)

    
documents = document_loader()
doc_chunks = split_documents(documents)
# print("Split ", len(documents), "into ", len(doc_chunks), "chunks.")

#  calculating chunk_id in the format source:page:page_repetition
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
        print(chunk.metadata)

calculate_chunk_id(doc_chunks)
        
        


        