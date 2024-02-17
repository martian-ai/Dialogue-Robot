import os
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.llms import ChatGLM
from langchain_community.llms import ChatGLM
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


import sys
sys.path.append("/home/disk2/sunhongchao/baidu/BotMVP4")
from solutions.prompt_collection import prompt_qa


embedding_mode_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-tiny",
    "text2vec3-online": "shibing624/text2vec-base-chinese",
    "text2vec3": "resources/embedding/text2vec3"
}


def load_documents(file_path="./三体2.txt"):
    """
    从指定目录加载文档并返回一个Document对象的列表。
    
    Args:
        无参数。
    
    Returns:
        一个Document对象的列表。
    
    """
    # loader = DirectoryLoader('三体', glob="*.py", show_progress=True, use_multithreading=True)
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    text_spliter = CharacterTextSplitter(chunk_size=520, 
                                         chunk_overlap=32,
                                         separator='\n')
    split_docs = text_spliter.split_documents(documents)

    print(len(split_docs))
    print(split_docs[:1])

    return split_docs


def load_embedding_model(model_name):
    """
    加载预训练的词嵌入模型
    
    Args:
        model_name (str): 预训练模型名称
    
    Returns:
        HuggingFaceEmbeddings: 加载的预训练词嵌入模型
    """
    encode_kwargs = {"normalize_embeddings":False}
    model_kwargs = {"device": "cuda:3"}
    return HuggingFaceEmbeddings(model_name=embedding_mode_dict[model_name],
                                 model_kwargs=model_kwargs,
                                 encode_kwargs=encode_kwargs)


def store_chroma(docs, embeddings, persist_directory):
    """
    将文档和嵌入存储为Chroma数据库。
    
    Args:
        docs (List[str]): 文档列表。
        embeddings (List[np.ndarray]): 嵌入列表，每个嵌入是numpy数组。
        persist_directory (str, optional): 持久化目录。默认为'VectorStore'。
    
    Returns:
        Chroma: Chroma数据库对象。
    """
    print(len(docs))
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def get_doc_qa_answer(query="", context=""):

    embeddings = load_embedding_model('text2vec3')
    print("end load embedding")

    embeddings = load_embedding_model('text2vec3')
    VectorStore = 'VectorStore302'
    if not os.path.exists(VectorStore):
        documents = load_documents("resources/corpus/service_orqa/document/三体.txt")
        print("end load documents")
        db = store_chroma(documents, embeddings, persist_directory=VectorStore)
    else:
        db = Chroma(persist_directory=VectorStore, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    
    llm = ChatGLM(
        endpoint_url='http://127.0.0.1:8000',
        max_token=8000,
        top_p=0.9
    )

    chain_type_kwargs = {"prompt": prompt_qa}
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
    
    while True:
        query = input("\n请输入问题: ")
        if query == "exit":
            break
    
        print(query)
        context = retriever.get_relevant_documents(query)
        print(context)
        # res = chain({"query":query})
        res = chain(query)
        answer, docs = res['result'], res['source_documents']
 
        print("\n\n> 问题:")
        print(query)
        print("\n> 回答:")
        print(answer)
 
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")


if __name__ == '__main__':
    get_doc_qa_answer()