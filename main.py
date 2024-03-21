import asyncio
import asyncpg
from google.cloud.alloydb.connector import AsyncConnector, IPTypes
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBLoader,
    AlloyDBVectorStore,
    Column,
)
from langchain_google_vertexai import VertexAIEmbeddings
from sqlalchemy.ext.asyncio import create_async_engine
import uuid

# AlloyDB info
PROJECT_ID = "duwenxin-space"
REGION = "us-central1"  # @param {type:"string"}
CLUSTER_NAME = "my-alloydb-cluster"  # @param {type:"string"}
INSTANCE_NAME = "my-alloydb-instance"  # @param {type:"string"}
DATABASE_NAME = "netflix"  # @param {type:"string"}
USER = "postgres"  # @param {type:"string"}
PASSWORD = "postgres"  # @param {type:"string"}

source_table_name = "wine_reviews"
vector_table_name = "wine_review_vector"

# Dataset
dataset_path = "wine_reviews_dataset.csv"
dataset_columns = [
    "country",
    "description",
    "designation",
    "points",
    "price",
    "province",
    "region_1",
    "region_2",
    "taster_name",
    "taster_twitter_handle",
    "title",
    "variety",
    "winery",
]

connection_string = f"projects/{PROJECT_ID}/locations/{REGION}/clusters/{CLUSTER_NAME}/instances/{INSTANCE_NAME}"
# initialize Connector object
connector = AsyncConnector()
engine = None


async def load_dataset_into_database():
    global engine
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=CLUSTER_NAME,
        instance=INSTANCE_NAME,
        database=DATABASE_NAME,
    )
    print("Successfully connected to AlloyDB database.")

    loader = await AlloyDBLoader.create(
        engine=engine,
        query=f"SELECT * FROM {source_table_name};",
        content_columns=dataset_columns,
    )

    documents = await loader.aload()
    print("Documents loaded.")
    return documents


async def create_vector_store_table(documents):
    print("Initializaing Vectorstore tables...")
    await engine.ainit_vectorstore_table(
        vector_size=1000,
        metadata_columns=[
            Column("country", "VARCHAR", nullable=True),
            Column("description", "VARCHAR", nullable=True),
            Column("designation", "VARCHAR", nullable=True),
            Column("points", "VARCHAR", nullable=True),
            Column("price", "INTEGER", nullable=True),
            Column("province", "VARCHAR", nullable=True),
            Column("region_1", "VARCHAR", nullable=True),
            Column("region_2", "VARCHAR", nullable=True),
            Column("taster_name", "VARCHAR", nullable=True),
            Column("taster_twitter_handle", "VARCHAR", nullable=True),
            Column("title", "VARCHAR", nullable=True),
            Column("variety", "VARCHAR", nullable=True),
            Column("winery", "VARCHAR", nullable=True),
        ],
        overwrite_existing=True,  # Enabling this will recreate the table if exists.
    )
    embedding = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest", project=PROJECT_ID
    )

    # Initialize AlloyDBVectorStore
    print("Initializing VectorStore...")
    vector_store = await AlloyDBVectorStore.create(
        engine=engine,
        table_name=vector_table_name,
        embedding_service=embedding,
        metadata_columns=dataset_columns,
    )

    ids = [str(uuid.uuid4()) for i in range(100)]
    vector_store.add_documents(documents, ids)
    print("Vector table created.")


async def main():
    documents = await load_dataset_into_database()
    await create_vector_store_table(documents)


if __name__ == "__main__":
    asyncio.run(main())
