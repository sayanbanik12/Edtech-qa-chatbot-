import os
import streamlit as st
from qa_chain import create_vector_db, get_qa_chain

st.title("Chat with AI 🌱")

# Ensure FAISS index exists
vectordb_file_path = "faiss_index"
if not os.path.exists(vectordb_file_path):
    st.warning("Vector DB not found. Click the button below to create it.")
    if st.button("Create Knowledgebase"):
        create_vector_db()
        st.success("Knowledge base created successfully!")
else:
    st.info("Knowledge base already exists. You can ask questions now!")

# Input for user question
question = st.text_input("Ask a question:")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"query": question})  # RetrievalQA expects dict
    st.header("Answer")
    st.write(response["result"])
