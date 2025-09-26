import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# LLM import
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings and vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

# Create vector DB
def create_vector_db():
    loader = CSVLoader(file_path="prompt_data.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)
    print("✅ Vector database created.")

# RetrievalQA chain
def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True  # required for FAISS index reload
    )
    retriever = vectordb.as_retriever(search_kwargs={"score_threshold": 0.7})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document without making unnecessary changes.
    If the answer is not found in the context, kindly state "I don't know." Do not try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Main execution
if __name__ == "__main__":
    if not os.path.exists(vectordb_file_path):
        create_vector_db()
    else:
        print("ℹ️ Vector DB already exists, skipping creation.")

    chain = get_qa_chain()
    query = "Do you have javascript course?"
    result = chain.invoke({"query": query})  # RetrievalQA expects dict input
    print("Query:", query)
    print("Answer:", result["result"])
