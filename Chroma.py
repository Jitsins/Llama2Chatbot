import json
import uuid
import chromadb
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA


directory = './input'


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


documents = load_docs(directory)
docs = split_docs(documents)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
unique_ids = list(set(ids))

# Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
seen_ids = set()
unique_docs = [doc for doc, id in zip(
    docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]


persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=unique_docs, ids=unique_ids, embedding=embeddings, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
vectordb.persist()

query = "Date of the State of Union address?"

new_db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)


matching_docs = new_db.similarity_search_with_score(query)


llm = LlamaCpp(
    n_gpu_layers=-1,
    n_batch=2048,
    n_ctx=2048,
    model_path=f"C:\My Drive\Projects\AI\Projects\ChatBot\models\llama-2-7b-chat.Q2_K.gguf",
    input={"temperature": 0.1,
           "max_length": 2000,
           "top_p": 1
           },
    verbose=True,  # True
)

docs = [Document(page_content=result[0].page_content,
                 metadata=result[0].metadata) for result in matching_docs]
# docs = [Document(page_content=matching_docs[0][0].page_content,
#                  metadata=matching_docs[0][0].metadata)]


# chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
# answer = chain.run(input_documents=docs, question=query)

retriever = new_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2})
# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
result = qa({"query": query})
print(result.result)

# print(answer)
