import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore, vectorstores

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vectorstores = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstores.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based one the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manually step-by-step execution.
    - No build-in streaming support.
    - No async support without additional code.
    - Harder to compose with other chains.
    - More verbose and error-prone.
    """
    
    # Step 1: Retrieve relevant documents.
    docs = retriever.invoke(query)

    # Step 2: Format the documents into context string.
    context = format_docs(docs)

    # Step 3: Format the prompt with the context and question.
    messages = prompt_template.invoke({"context": context, "question": query})

    # Step 4: Invoke LLM with the formatted messages.
    response = llm.invoke(messages)

    # Step 5: Return the response content.
    return response.content

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving...")

    #Query
    query = "What is Pinecoin in machine learning?"
    
    # Option 1: Without LCEL
    print("\n" + "="*70)
    print("Option 1: Without LCEL")
    print("="*70)
    result_without_lcel  = retrieval_chain_without_lcel(query)
    print("Answer:")
    print(result_without_lcel)

