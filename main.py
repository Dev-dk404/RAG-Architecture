import data_loader
import vector_database
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate

def prompt_creator(results, query):
    template = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}

        """
    prompt_template = PromptTemplate.from_template(template)
    context_text = "\n\n ---- \n\n".join([result.page_content for result, _score in results])
    return prompt_template.format(context = context_text, question = query)


def main():
    # documents = data_loader.load_data()
    # split_docs = data_loader.split_documents(documents)

    # chunks = data_loader.calculate_chunk_id(split_docs)

    # # print(chunks[0].metadata["id"])
    # vector_database.insert_into_db(chunks)

    vector_store = vector_database.load_local_db()
    results = vector_store.similarity_search("How did Goodfellow describe adverserial nets?", k=3)


    model = Ollama(model= 'mistral')
    prompt = prompt_creator(results, "How did Goodfellow describe adverserial nets?")
    response = model.invoke(prompt)
    print(response)

if __name__ == "__main__":
    main()

