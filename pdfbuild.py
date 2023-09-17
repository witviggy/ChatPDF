import streamlit as st
from dotenv import load_dotenv
import pickle
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
    st.title('🤖💬 LLM Chat App')
    st.markdown('''
    ## About
    Welcome to the LLM-powered chatbot! This chatbot is built using cutting-edge technologies:
    - [Streamlit](https://streamlit.io/) 🚀
    - [LangChain](https://python.langchain.com/) 📚
    - [OpenAI](https://platform.openai.com/docs/models) LLM model 🧠

    Feel free to explore its features and ask questions about your PDF files!
    ''')

    # Add a section for user options
    st.header("🛠️ User Options")
    enable_embeddings = st.checkbox("Enable Embeddings", value=True)
    enable_chat_history = st.checkbox("Show Chat History", value=True)

    st.write("Customize your chatbot experience:")
    st.write("- Enable or disable embeddings for text processing.")
    st.write("- Choose to show or hide the chat history.")

    st.header("👤 User Information")
    user_name = st.text_input("Your Name", "")
    user_email = st.text_input("Your Email", "")

    st.write("Share your information with the chatbot (optional):")
    st.write("- Personalize your interactions.")
    st.write("- Receive responses and recommendations.")

    st.header("🔗 Links and Credits")
    st.markdown("[Prompt Engineer YouTube](https://youtube.com/@engineerprompt)")
    st.markdown("[GitHub Repository](https://github.com/yourusername/yourrepository)")
    st.markdown("[Report an Issue](https://github.com/yourusername/yourrepository/issues)")

    st.write("Explore more:")
    st.write("- Watch tutorial videos.")
    st.write("- Contribute to the open-source project.")
    st.write("- Report issues or provide feedback.")
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

        # Define your OpenAI API key here
        openai_api_key = "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"

        # Define store_name as you did previously
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

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
