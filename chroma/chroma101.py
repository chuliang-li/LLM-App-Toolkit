import chromadb
from pprint import pprint
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


chroma_client = chromadb.PersistentClient("./testdb")

chroma_client.delete_collection("my_collection")

embedding_functions = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh"
)

collection = chroma_client.create_collection(
    name="my_collection",
    embedding_function=embedding_functions
    )

collection.add(
    ids=["id1", "id2"],
    documents=[
        "苹果是水果",
        "梨子是水果"

    ]
)

results = collection.query(
    query_texts=["橙子是水果"], # Chroma will embed this for you
    n_results=2, # how many results to return
    include= ["metadatas","documents","distances"]

)
pprint(results)
