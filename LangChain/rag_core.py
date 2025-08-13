# rag_core.py
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Removed imports related to document loading, splitting, and Chroma creation
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma


def get_dashscope_api_key():
    """
    Retrieves the DashScope API key from environment variables.
    Raises an error if the key is not set.
    """
    load_dotenv()
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set. Please set your Alibaba Cloud DashScope API key.")
    return dashscope_api_key

# Removed get_vector_store function as its logic moved to data_preparation.py

def get_rag_chain(vectorstore):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using the
    Aliyun Tongyi Qianwen model. It now directly receives the vectorstore.
    """
    dashscope_api_key = get_dashscope_api_key()
    llm = ChatOpenAI(
        model_name="qwen3-coder-30b-a3b-instruct",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=dashscope_api_key,
        temperature=0
    )

    template = """
    你是一个有用的问答助手。请根据提供的上下文信息来回答问题。
    如果问题无法从上下文中找到答案，请说你不知道。

    上下文:
    {context}

    问题: {question}
    有帮助的答案:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

def get_no_rag_chain():
    """
    Creates a pure LLM chain without retrieval, using the Aliyun Tongyi Qianwen model.
    """
    dashscope_api_key = get_dashscope_api_key()
    llm = ChatOpenAI(
        model_name="qwen3-coder-30b-a3b-instruct",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=dashscope_api_key,
        temperature=0
    )
    return llm