import asyncio
from langchain_google_alloydb_pg import AlloyDBVectorStore
from langchain_google_alloydb_pg.indexes import IVFFlatIndex
from langchain_google_alloydb_pg.indexes import HNSWIndex
from create_vector_embeddings import (
    PROJECT_ID,
    REGION,
    CLUSTER_NAME,
    INSTANCE_NAME,
    DATABASE_NAME,
    USER,
    PASSWORD,
    vector_table_name,
)
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBLoader,
    AlloyDBVectorStore,
    Column,
)
from langchain_google_vertexai import VertexAIEmbeddings

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project=PROJECT_ID
)


async def get_vector_store():
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=CLUSTER_NAME,
        instance=INSTANCE_NAME,
        database=DATABASE_NAME,
        user=USER,
        password=PASSWORD,
    )

    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
    )
    return vector_store


async def hnsw_search(vector_store):
    hnsw_index = HNSWIndex()
    await vector_store.aapply_vector_index(hnsw_index)
    query = "Aromas include tropical fruit, broom, brimstone and dried herb."
    docs = await vector_store.asimilarity_search(query, k=10)
    await vector_store.adrop_vector_index(hnsw_index.name)
    return docs


async def ivfflat_search(vector_store):
    ivfflat_index = IVFFlatIndex()
    await vector_store.aapply_vector_index(ivfflat_index)
    query = "Aromas include tropical fruit, broom, brimstone and dried herb."
    docs = await vector_store.asimilarity_search(query, k=10)
    await vector_store.adrop_vector_index(ivfflat_index.name)
    return docs


async def knn_search(vector_store):
    query = "Aromas include tropical fruit, broom, brimstone and dried herb."
    docs = await vector_store.asimilarity_search(query, k=10)
    return docs


def calculate_recall(base, target) -> float:
    # size of intersection / total number of times
    match = 0
    total = len(base)
    for i in range(total):
        if base[i].metadata["id"] == target[i].metadata["id"]:
            match = match + 1
    return match / total


async def main():
    vector_store = await get_vector_store()
    knn_docs = await knn_search(vector_store)
    hnsw_docs = await hnsw_search(vector_store)
    ivfflat_docs = await ivfflat_search(vector_store)
    hnsw_recall = calculate_recall(knn_docs, hnsw_docs)
    ivfflat_recall = calculate_recall(knn_docs, ivfflat_docs)
    print(f"HNSW recall: {hnsw_recall}")
    print(f"IVFFLAT recall: {ivfflat_recall}")


if __name__ == "__main__":
    asyncio.run(main())
