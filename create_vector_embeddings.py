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
import pandas as pd
import uuid
import sqlalchemy
import numpy as np


EMBEDDING_COUNT = 5000

# AlloyDB info
PROJECT_ID = "duwenxin-space"
REGION = "us-central1"  # @param {type:"string"}
CLUSTER_NAME = "my-alloydb-cluster"  # @param {type:"string"}
INSTANCE_NAME = "my-alloydb-instance"  # @param {type:"string"}
DATABASE_NAME = "netflix"  # @param {type:"string"}
USER = "postgres"  # @param {type:"string"}
PASSWORD = "postgres"  # @param {type:"string"}

source_table_name = "wines"
vector_table_name = "wines_vector"

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


async def getconn():
    conn = await connector.connect(
        connection_string,
        "asyncpg",
        user=USER,
        password=PASSWORD,
        db=DATABASE_NAME,
        enable_iam_auth=False,
        ip_type=IPTypes.PUBLIC,
    )
    return conn


# create connection pool
pool = create_async_engine(
    "postgresql+asyncpg://", async_creator=getconn, isolation_level="AUTOCOMMIT"
)


async def import_data_to_alloydb():
    df = pd.read_csv(dataset_path)

    delete_table_cmd = sqlalchemy.text(f"""DROP TABLE IF EXISTS {source_table_name};""")

    create_table_cmd = sqlalchemy.text(
        f"""CREATE TABLE {source_table_name} (
            id SERIAL PRIMARY KEY,
            country VARCHAR(100),
            description TEXT,
            designation VARCHAR(255),
            points INT,
            price NUMERIC(10, 2),
            province VARCHAR(100),
            region_1 VARCHAR(100),
            region_2 VARCHAR(100),
            taster_name VARCHAR(100),
            taster_twitter_handle VARCHAR(100),
            title TEXT,
            variety VARCHAR(100),
            winery VARCHAR(100)
        );"""
    )

    insert_data_cmd = sqlalchemy.text(
        f"""
        INSERT INTO {source_table_name} VALUES (:id, :country, :description, :designation,
            :points, :price, :province, :region_1, :region_2, :taster_name,
            :taster_twitter_handle, :title, :variety, :winery)
        """
    )

    parameter_map = [
        {
            "id": index,
            "country": row["country"] if not pd.isna(row["country"]) else None,
            "description": (
                row["description"] if not pd.isna(row["description"]) else None
            ),
            "designation": (
                row["designation"] if not pd.isna(row["designation"]) else None
            ),
            "points": row["points"] if not pd.isna(row["points"]) else None,
            "price": row["price"] if not pd.isna(row["price"]) else None,
            "province": row["province"] if not pd.isna(row["province"]) else None,
            "region_1": row["region_1"] if not pd.isna(row["region_1"]) else None,
            "region_2": row["region_2"] if not pd.isna(row["region_2"]) else None,
            "taster_name": (
                row["taster_name"] if not pd.isna(row["taster_name"]) else None
            ),
            "taster_twitter_handle": (
                row["taster_twitter_handle"]
                if not pd.isna(row["taster_twitter_handle"])
                else None
            ),
            "title": row["title"] if not pd.isna(row["title"]) else None,
            "variety": row["variety"] if not pd.isna(row["variety"]) else None,
            "winery": row["winery"] if not pd.isna(row["winery"]) else None,
        }
        for index, row in df.iterrows()
    ]

    async with pool.connect() as db_conn:
        await db_conn.execute(delete_table_cmd)
        await db_conn.execute(create_table_cmd)
        await db_conn.execute(
            insert_data_cmd,
            parameter_map,
        )
        await db_conn.commit()
    await connector.close()


async def load_alloydb_documents():
    global engine
    engine = await AlloyDBEngine.afrom_instance(
        project_id=PROJECT_ID,
        region=REGION,
        cluster=CLUSTER_NAME,
        instance=INSTANCE_NAME,
        database=DATABASE_NAME,
        user=USER,
        password=PASSWORD,
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
        table_name=vector_table_name,
        vector_size=768,
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

    ids = [str(uuid.uuid4()) for i in range(EMBEDDING_COUNT)]
    await vector_store.aadd_documents(documents, ids)
    print("Vector table created.")


async def main():
    await import_data_to_alloydb()
    documents = await load_alloydb_documents()
    await create_vector_store_table(documents)


if __name__ == "__main__":
    asyncio.run(main())
