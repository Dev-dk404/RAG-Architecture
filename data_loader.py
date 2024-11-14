from langchain.document_loaders import PyPDFLoader

def document_loader():
    loader = PyPDFLoader("data/marriage-rule-book.pdf")
    pages = loader.load_and_split()
    return pages

pages = document_loader()

for document in pages:
    print()
    print(document, "\n")