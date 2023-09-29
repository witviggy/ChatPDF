import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤖💬PDFChat')
    st.markdown('''
    ## About
    "Welcome to the LLM-powered chatbot! This chatbot is built using cutting-edge technologies:

    Streamlit 🚀: Streamlit is a fast and efficient Python library for creating interactive web applications. It allows us to build this chatbot with ease.
                
    LangChain 📚: LangChain is a Python library that provides tools for natural language processing tasks like text splitting and embeddings.
                
    OpenAI LLM model 🧠: OpenAI's LLM (Language Model) is a powerful AI model that understands and generates human-like text. It powers the intelligence behind this chatbot.

    Feel free to explore its features and ask questions about your PDF files! This chatbot leverages these technologies to provide you with information and answers related to PDF documents."
    ''')

    # Add a section for user options
    st.header("🛠️ User Options")
    enable_embeddings = st.checkbox("Enable Embeddings", value=True)
    enable_chat_history = st.checkbox("Show Chat History", value=True)

    st.write("Customize your chatbot experience:")
    st.write("- Enable or disable embeddings for text processing.")
    st.write("- Choose to show or hide the chat history.")

    st.header("🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/witviggy/ChatPDF)")
    st.markdown("[Report an Issue](https://github.com/witviggy/ChatPDF/issues)")

    add_vertical_space(5)
    st.write('Made with ❤️ by Vignesh M')

def main():
    st.header("Chat with PDF 💬")

    load_dotenv()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Define store_name as you did previously
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # Load the OpenAI API key from the environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if enable_embeddings:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        else:
            VectorStore = FAISS.from_texts(chunks)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
