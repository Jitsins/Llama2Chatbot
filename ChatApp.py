from tabnanny import verbose
import uuid
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sympy import true
from htmlTemplates import css, bot_template, user_template
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceHub
import wikipediaapi
from streamlit_modal import Modal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.docstore.document import Document


# model_path = "C:\My Drive\Projects\AI\Projects\ChatBot\models\llama-2-7b-chat.Q2_K.gguf"
model_path = "C:\My Drive\Projects\AI\Projects\ChatBot\models\llama2_7b_chat_uncensored.Q4_K_M.gguf"
file_directory_paths = "./tempFiles"
persist_directory = "chroma_db"


def load_pdfs_create_docs(pdf_docs):
    for pdf_doc in pdf_docs:
        bytes_data = pdf_doc.read()
        with open(f"{file_directory_paths}/{pdf_doc.name}", "wb") as file:
            file.write(bytes_data)

    # loader = DirectoryLoader(file_directory_paths)
    loader = PyPDFDirectoryLoader(file_directory_paths)
    documents = loader.load()
    st.write(documents)
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# def get_text_chunks(text, chunk_size=1000, chunk_overlap=20):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     docs = text_splitter.split_documents(documents)
#     return docs


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = embeddings = LlamaCppEmbeddings(model_path=model_path,
                                                 n_ctx=1000,
                                                 n_gpu_layers=20,
                                                 n_batch=512)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"Temperature":0.3, "max_length":2048})
    llm = LlamaCpp(
        model_path=model_path,
        input={"temperature": 0.1,
               "max_length": 1000,
               "top_p": 1
               },
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,  # True
        n_ctx=1000,
        n_gqa=8,
        use_mlock=True,
        n_gpu_layers=20,
        n_batch=512
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def getWikiText(keyword):

    # Create a Wikipedia API object
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org) generic-library/0.0',
                                       language='en',
                                       extract_format=wikipediaapi.ExtractFormat.WIKI)  # 'en' for English Wikipedia, you can change it to the desired language code

    # Fetch the page summary
    page = wiki_wiki.page(keyword)
    
    # Print the page summary
    return [Document(page_content=page.text, metadata={"source": f"wiki/{keyword}"})]


def load_docs(option, control):
    match option:
        case "Wiki":
            if not control:
                st.toast("Enter a valid keyword", icon='🔥')
            else:
                return getWikiText(control)
        case "File":
            if not control:
                st.toast("Drop atleast 1 file", icon='🔥')
            else:
                return load_pdfs_create_docs(control)


def get_llm():
    return LlamaCpp(
        n_gpu_layers=-1,
        n_batch=2048,
        n_ctx=2048,
        model_path=f"models\llama2_7b_chat_uncensored.Q4_K_M.gguf",
        input={"temperature": 0.1,
               "max_length": 2000,
               "top_p": 1
               },
        verbose=False,  # True
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    # Get LLM
    if 'llm' in st.session_state:
        llm = st.session_state.llm
    else:
        llm = get_llm()
        st.session_state.llm = llm

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    form = st.form(key='my-form')
    name = form.text_area('Ask a question about your documents:')
    submit = form.form_submit_button('Submit')

    if submit:
        # handle_userinput(user_question)
        new_db = Chroma(persist_directory=persist_directory,
                        embedding_function=st.session_state.embeddings)

        retriever = new_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 4})
        # create a chain to answer questions
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        st.write(qa({"query": name}))

    with st.sidebar:
        st.subheader("Enter Keyword to look for in Wiki")
        radio = st.radio("Select Mode", ("Wiki", "File"))
        if (radio == 'Wiki'):
            control = st.text_input("Wiki Keyword")
        elif (radio == 'File'):
            control = st.file_uploader(
                "Choose PDF file(s) for upload", accept_multiple_files=True)
        if st.button("Process"):
            documents = load_docs(radio, control)
            st.write(documents)

            # Split docs to fit in chunks
            docs = split_docs(documents)

            # Create Embeddings
            embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2")
            st.session_state.embeddings = embeddings

            # Create Unique Ids
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
                   for doc in docs]
            unique_ids = list(set(ids))

            # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
            seen_ids = set()
            unique_docs = [doc for doc, id in zip(
                docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

            vectordb = Chroma.from_documents(
                documents=unique_docs, ids=unique_ids, embedding=embeddings, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
            vectordb.persist()


if __name__ == '__main__':
    main()
