from milvus import default_server
from pymilvus import connections, utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
import gdown
import zipfile
import warnings
import datetime
warnings.filterwarnings("ignore")

default_server.set_base_dir('milvus_data')
default_server.cleanup()
default_server.start()
connections.connect(host='127.0.0.1', port=default_server.listen_port)


import pandas as pd
df = pd.read_csv("./Prevent/DiabetesStudy.csv")

cleaned_df = df.dropna()
#df.to_csv("./Prevent/cleanedData.csv",index=False)

COLLECTION_NAME = "DiabetesStudy"
BATCH_SIZE = 128
DIMENSION=14
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
   FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
   FieldSchema(name="Patient_ID", dtype=DataType.DOUBLE),
   FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
   
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
   "index_type": "IVF_FLAT",
   "metric_type": "L2",
   "params": {"nlist": 128},
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

def embed_insert(data: list):
    ins = [
       data[0],
       data[1]

   ]
    collection.insert(ins)
    print("Batch Inserted")


data_batch = [[],[]]
for index, row in cleaned_df.iterrows():
    pid=row["Patient_ID"]
    row=row.drop("Patient_ID")
    kt= row.to_numpy()
    data_batch[0].append(pid)
    data_batch[1].append(kt)
    if len(data_batch[0]) % BATCH_SIZE == 0:
        embed_insert(data_batch)
        data_batch = [[],[]]

if len(data_batch[0]) != 0:
    embed_insert(data_batch)

collection.flush()














