# data_preparation.py
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb # Import chromadb for direct client interaction if needed, though LangChain wrappers handle most.

def generate_rag_data(file_path="knowledge_base.txt"):
    """
    生成用于RAG的更专业、更具区分度的示例知识库数据。
    聚焦于“未来计算架构”领域。
    """
    data = r"""
    ## 未来计算架构的演进

    **1. 量子计算 (Quantum Computing):**
    量子计算利用量子力学现象（如叠加和纠缠）来执行计算。与传统位（0或1）不同，量子位（qubits）可以同时表示0和1。主要的量子计算模型包括门模型（Gate-based Quantum Computing）、退火模型（Quantum Annealing）和拓扑量子计算（Topological Quantum Computing）。IBM的Qiskit和Google的Cirq是流行的量子编程框架。量子霸权（Quantum Supremacy）指的是量子计算机在特定任务上超越最强大的经典计算机的能力。

    **2. 神经形态计算 (Neuromorphic Computing):**
    神经形态计算旨在模仿人脑的结构和功能，以实现更高的能效和并行处理能力。其核心组件是“神经元”和“突触”，它们可以直接在硬件层面模拟生物神经网络的行为。IBM的TrueNorth芯片是神经形态计算的一个早期代表，而英特尔的Loihi系列芯片则进一步推动了这一领域的发展，专注于事件驱动的稀疏处理。这种架构特别适用于人工智能和边缘计算场景。

    **3. 光子计算 (Photonic Computing):**
    光子计算使用光子（而非电子）来传输和处理信息。由于光子速度快、能耗低且不受电磁干扰，光子计算在某些特定任务（如高速数据传输、模拟计算和线性代数运算）中展现出巨大潜力。硅光子学技术是实现集成光子计算的关键，它允许在硅基芯片上集成光学元件。

    **4. 类脑计算 (Brain-inspired Computing):**
    类脑计算是一个更广泛的概念，它不仅限于硬件层面模仿大脑结构，还包括算法和软件层面借鉴大脑的学习和认知机制。这与神经形态计算有所重叠，但更强调算法层面的创新，如脉冲神经网络（Spiking Neural Networks, SNNs）和认知计算模型。

    **5. 边缘计算与雾计算 (Edge and Fog Computing):**
    随着物联网(IoT)设备的普及，将数据处理能力推向数据源附近的边缘设备变得至关重要。边缘计算减少了数据传输的延迟和带宽需求，提高了响应速度和安全性。雾计算是边缘计算的一种延伸，它在云端和边缘设备之间引入了一个中间层，提供更广泛的计算、存储和网络服务。

    **6. 可逆计算 (Reversible Computing):**
    可逆计算是一种计算范式，其计算过程在逻辑上是可逆的，这意味着它在计算过程中不损失任何信息。理论上，可逆计算可以在不产生热量的情况下执行，这对于超低功耗计算和避免物理限制至关重要。Landauer原理指出，每次擦除一位信息至少会耗散 $kT \ln 2$ 的能量，可逆计算旨在规避这一限制。

    **7. 量子AI (Quantum AI):**
    量子AI是量子计算与人工智能的交叉领域。它利用量子算法来加速机器学习任务（如量子机器学习、量子优化）或模拟复杂的神经网络。典型的应用包括量子支持向量机 (QSVM) 和量子神经网络 (QNN)。

    **通用概念补充:**
    * **LangChain**: 一个用于开发由语言模型驱动的应用程序的框架，它提供了构建复杂LLM应用所需的组件和接口。
    * **Streamlit**: 用于快速构建和分享数据应用的Python库，使开发者能够用纯Python代码创建交互式Web应用。
    * **ChromaDB**: 一个开源的嵌入式数据库，用于存储和检索嵌入向量，常与语言模型结合实现高效语义搜索和RAG功能。
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)
    print(f"✅ 知识库数据已生成并保存到 {file_path}")

def _get_embedding_function():
    """Helper to load the embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ HuggingFace Embedding 模型 (BAAI/bge-small-zh) 加载成功。")
        return embeddings
    except Exception as e:
        print(f"❌ 嵌入模型加载失败: {e}")
        print("请确保已安装 'sentence-transformers' 库，并且网络连接正常以下载模型。")
        raise

def load_and_vectorize_data(file_path="knowledge_base.txt", persist_directory="./chroma_db", force_rebuild=False):
    """
    从文本文件加载数据，分割，并创建或加载Chroma向量存储。
    force_rebuild=True 会强制删除现有向量存储并重新创建。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 知识库文件 '{file_path}' 不存在。请先运行 generate_rag_data。")

    embeddings = _get_embedding_function()
    vectorstore = None
    
    # Decide if we should try to load or if we must rebuild
    # We rebuild if force_rebuild is True OR if the directory doesn't exist/is empty
    should_rebuild = force_rebuild or \
                     not os.path.exists(persist_directory) or \
                     len(os.listdir(persist_directory)) == 0

    if should_rebuild:
        # If rebuilding, first clean up any existing directory
        if os.path.exists(persist_directory):
            print(f"🔄 检测到需要重建向量存储，正在删除旧目录: {persist_directory}")
            shutil.rmtree(persist_directory)
            print("✅ 旧目录删除成功。")

        print(f"🔄 正在从 {file_path} 读取文档并创建新的向量存储...")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"✅ 已加载 {len(documents)} 个文档。")
        if documents:
            print(f"   第一个文档内容（前200字符）:\n---START---\n{documents[0].page_content[:200]}...\n---END---")
        else:
            print("   ⚠️ 未加载到任何文档内容！请检查 'knowledge_base.txt' 文件。")
            raise ValueError("知识库文件为空或无法加载。")

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        print(f"✅ 文档分割后生成了 {len(docs)} 个文本块。")
        if docs:
            print(f"   第一个文本块内容（前200字符）:\n---START---\n{docs[0].page_content[:200]}...\n---END---")
        else:
            print("   ❌ 未生成任何文本块！请检查文本分割器配置或文档内容。")
            raise ValueError("文本分割失败，未生成任何文本块。")

        print("🔄 正在创建 Chroma 向量存储并生成嵌入向量...")
        # IMPORTANT: This is the ONLY place Chroma.from_documents is called.
        # It creates the DB and implicitly persists it to persist_directory.
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"} # Match your insert_Vector.py
        )
        print(f"✅ Chroma 向量存储创建成功。")

        # Call persist() directly after from_documents.
        # While from_documents often implicitly persists, explicit call ensures consistency.
        vectorstore.persist()
        print(f"✅ 向量存储已持久化到 {persist_directory}。总计 {vectorstore._collection.count()} 个条目。")
        
    else: # should_rebuild is False, so try to load existing
        print(f"✅ 检测到 {persist_directory} 文件夹已存在，尝试从持久化数据加载。")
        try:
            # IMPORTANT: This is the ONLY place Chroma is loaded (not created from documents).
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            count = vectorstore._collection.count()
            if count > 0:
                print(f"✅ 向量存储已从持久化数据加载。总计 {count} 个条目。")
            else:
                # This scenario (directory exists but empty) implies corruption or incomplete write
                print(f"⚠️ {persist_directory} 文件夹存在但为空，或数据不完整。推荐重新生成。")
                # Fallback: force a rebuild if loaded DB is empty
                return load_and_vectorize_data(file_path, persist_directory, force_rebuild=True) 
        except Exception as e:
            print(f"❌ 从持久化数据加载 ChromaDB 失败: {e}。推荐重新生成。")
            # Fallback: force a rebuild if loading fails
            return load_and_vectorize_data(file_path, persist_directory, force_rebuild=True)

    return vectorstore

if __name__ == "__main__":
    print("--- 正在执行 data_preparation.py ---")
    generate_rag_data()
    try:
        # Running directly, you might want to force_rebuild=True for initial setup
        vectorstore_test = load_and_vectorize_data(force_rebuild=True) # Force rebuild for testing
        print("\n--- data_preparation.py 执行完成 ---")
    except Exception as e:
        print(f"\n❌ data_preparation.py 向量化过程中发生错误: {e}")