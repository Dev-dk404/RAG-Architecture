import data_loader
import vector_database
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")

def prompt_creator(results, query):
    template = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}

        """
    prompt_template = PromptTemplate.from_template(template)
    context_text = "\n\n ---- \n\n".join([result.page_content for result in results])
    return prompt_template.format(context = context_text, question = query)


def main():
    # Get input from user on what they want to learn from document data
    user_input = input("What would you like to know? Please Ask:")
    query_text = str(user_input)

    # Load data
    documents = data_loader.load_data()

    # Split data into chunks of predefined size
    split_docs = data_loader.split_documents(documents)
    chunks = data_loader.calculate_chunk_id(split_docs)

    # Check and insert any new chunks into the db
    vector_database.insert_into_db(chunks)

    # Load from local vector storage
    vector_store = vector_database.load_local_db()
    
    # Find 3 most similar document chunks to the query
    results = vector_store.similarity_search(query_text, k=3)

    # Use Mistral LLM model to format the response for user. 
    model = Ollama(model= 'mistral')
    prompt = prompt_creator(results, query_text)
    response = model.invoke(prompt, max_tokens = 150, temperature = 0.7)
    sources = [result.metadata.get("id")for result in results]
    print(f"Query from user: \n{query_text}\n")
    formated_response = f"Response:\n{response}\n Sources:\n {sources}\n"
    print(formated_response)

if __name__ == "__main__":
    main()

