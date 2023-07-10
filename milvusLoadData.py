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

##url = 'https://drive.google.com/uc?id=10_sVL0UmEog7mczLedK5s1pnlDOz3Ukf'
##output = './white_house_2021_2022.zip'
##gdown.download(url, output)
##with zipfile.ZipFile("./white_house_2021_2022.zip","r") as zip_ref:
##    zip_ref.extractall("./white_house_2021_2022")
import pandas as pd
df = pd.read_csv("./white_house_2021_2022/The white house speeches.csv")
df = df.dropna()
cleaned_df = df.loc[(df["Speech"].str.len() > 50)]
cleaned_df["Speech"] = cleaned_df["Speech"].str.replace("\r\n", "")
cleaned_df["Date_time"] = pd.to_datetime(cleaned_df["Date_time"], format="%B %d, %Y")
##print(cleaned_df)
COLLECTION_NAME = "white_house_2021_2022"
DIMENSION = 384
BATCH_SIZE = 128
TOPK = 3
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
   FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
   FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
   FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=100),
   FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=200),
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
from sentence_transformers import SentenceTransformer
transformer = SentenceTransformer('all-MiniLM-L6-v2')

def embed_insert(data: list):
    embeddings = transformer.encode(data[3])
    ins = [
       data[0],
       data[1],
       data[2],
       [x for x in embeddings]
   ]
    collection.insert(ins)

data_batch = [[], [], [], []]
for index, row in cleaned_df.iterrows():
    data_batch[0].append(row["Title"])
    data_batch[1].append(str(row["Date_time"]))
    data_batch[2].append(row["Location"])
    data_batch[3].append(row["Speech"])
    if len(data_batch[0]) % BATCH_SIZE == 0:
        embed_insert(data_batch)
        data_batch = [[], [], [], []]

if len(data_batch[0]) != 0:
    embed_insert(data_batch)

collection.flush()












