import asyncio
import time

from create_vector_embeddings import (
    CLUSTER_NAME,
    DATABASE_NAME,
    INSTANCE_NAME,
    PASSWORD,
    PROJECT_ID,
    REGION,
    USER,
    vector_table_name,
)
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBVectorStore,
    Column,
)
from langchain_google_alloydb_pg.indexes import (
    HNSWIndex,
    IVFFlatIndex,
    DistanceStrategy,
)
from langchain_google_vertexai import VertexAIEmbeddings

k = 10
query_1 = "Brooding aromas of barrel spice"
query_2 = "Aromas include tropical fruit, broom, brimstone and dried herb."
query = query_1

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
    # Distance strategy: EUCLIDEAN, COSINE_DISTANCE, INNER_PRODUCT
    hnsw_index = HNSWIndex(
        distance_strategy=DistanceStrategy.INNER_PRODUCT, m=99, ef_construction=200
    )
    await vector_store.aapply_vector_index(hnsw_index)
    assert await vector_store.is_valid_index(hnsw_index.name)

    start = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end = time.monotonic()  # timer ends

    await vector_store.adrop_vector_index(hnsw_index.name)
    latency = round(end - start, 2)
    return docs, latency


async def ivfflat_search(vector_store):
    ivfflat_index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
    await vector_store.aapply_vector_index(ivfflat_index)
    assert await vector_store.is_valid_index(ivfflat_index.name)

    start = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end = time.monotonic()  # timer ends

    await vector_store.adrop_vector_index(ivfflat_index.name)
    latency = round(end - start, 2)
    return docs, latency


async def knn_search(vector_store):

    start = time.monotonic()  # timer starts
    docs = await vector_store.asimilarity_search(query, k=k)
    end = time.monotonic()  # timer ends

    latency = round(end - start, 2)
    return docs, latency


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
    knn_docs, knn_latency = await knn_search(vector_store)
    hnsw_docs, hnsw_latency = await hnsw_search(vector_store)
    ivfflat_docs, ivfflat_latency = await ivfflat_search(vector_store)
    hnsw_recall = calculate_recall(knn_docs, hnsw_docs)
    ivfflat_recall = calculate_recall(knn_docs, ivfflat_docs)

    print(f"KNN recall: 1.0            KNN latency: {knn_latency}")
    print(f"HNSW recall: {hnsw_recall}          HNSW latency: {hnsw_latency}")
    print(f"IVFFLAT recall: {ivfflat_recall}    IVFFLAT latency: {ivfflat_latency}")


if __name__ == "__main__":
    asyncio.run(main())
