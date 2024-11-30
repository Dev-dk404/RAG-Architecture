from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from data_embedding import get_embedding_function


DB_PATH = "vectorFAISS_db"
def insert_into_db(chunks :list[Document]):
    embeddings = get_embedding_function()
    db = FAISS.from_documents(documents = chunks, embedding = embeddings)

    db.save_local(DB_PATH)

def load_local_db():
    embeddings = get_embedding_function()
    db = FAISS.load_local(DB_PATH,embeddings, allow_dangerous_deserialization= True )
    return db
