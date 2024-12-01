from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from data_embedding import get_embedding_function


DB_PATH = "vectorFAISS_db"

def load_local_db():
    embeddings = get_embedding_function()
    db = FAISS.load_local(DB_PATH,embeddings, allow_dangerous_deserialization= True )
    return db

def insert_into_db(chunks :list[Document]):
    embeddings = get_embedding_function()

    # check if the chunk is already in the vector store, skip if true
    db = load_local_db()
    stored_chunks = db.docstore._dict
    existing_chunks = set([key for key, _ in stored_chunks.items()])
    new_chunks = []

    # Make a list of new documents to be added
    for chunk in chunks:
        chunk_id = chunk.metadata["id"]
        if chunk_id not in existing_chunks:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}\n")
        ids = [chunk.metadata["id"] for chunk in new_chunks]
        db = FAISS.from_documents(documents = new_chunks, embedding = embeddings, ids = ids)

        db.save_local(DB_PATH)
        print("✅ Done adding....\n")
    else:
        print("✅ No new documents to add to the vector store.\n")
