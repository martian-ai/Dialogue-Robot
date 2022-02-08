#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
# Description: 
# Author: apollo2mars apollo2mars@gmail.com
################################################################################

import os
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import chroma, Chroma

embedding_mode_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-tiny",
    # "text2vec3": "shibing624/text2vec-base-chinese"
    "text2vec3": "text2vec3"
}

print("begin")

def load_documents(directory='books'):
    # loader = DirectoryLoader('三体', glob="*.py", show_progress=True, use_multithreading=True)
    loader = TextLoader('三体.txt', encoding='utf-8')
    documents = loader.load()
    for document in documents:
        print(document)
    text_spliter = CharacterTextSplitter(chunk_size=64, 
                                         chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_embedding_mode(model_name='shibing624/text2vec-base/chinese'):
    encode_kwargs = {"normalize_embeddings":False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(model_name=embedding_mode_dict[model_name],
                                 model_kwargs=model_kwargs,
                                 encode_kwargs=encode_kwargs)

def store_chroma(docs, embeddings, persist_directory='VectorStore'):
    print(docs)
    print(embeddings)
    db = Chroma.from_documents(docs, embeddings, persist_directory)
    # db = Chroma
    db.persist()
    return db

documents = load_documents('books')
print("end load documents")

embeddings = load_embedding_mode('text2vec3')
print("end load embedding")

db = store_chroma(documents, embeddings)
print("end load db")

# if not os.path.exists("VectorStore"):
#     documents = load_documents()
#     print("end load documents")
#     db = store_chroma(documents, embeddings)
#     print("end load db")
# else:
#     db = chroma(persisi_distory="VectorStore", embedding_function=embeddings)

