from pymilvus import Collection, utility,connections
from milvus import default_server
connections.connect(host='127.0.0.1', port=default_server.listen_port)
collection = Collection("white_house_2021_2022")
collection.load()
utility.load_state("white_house_2021_2022")
utility.loading_progress("white_house_2021_2022")

TOPK = 3
import time
from sentence_transformers import SentenceTransformer
transformer = SentenceTransformer('all-MiniLM-L6-v2')
search_terms = ["The President speaks about the impact of renewable energy at the National Renewable Energy Lab.", "The Vice President and the Prime Minister of Canada both speak."]
def embed_search(data):
    embeds = transformer.encode(data)
    return [x for x in embeds]

search_data = embed_search (search_terms)


start = time.time()
res = collection.search(
    data= search_data,
    anns_field ="embedding",
    param={"metric_type": "L2",
           "params": {"nprobe":10}},
    limit = TOPK,
    output_fields = ["title"]
    )
end = time.time()

for hits_i, hits in enumerate(res):
    print("\nTitle: ", search_terms[hits_i])
    print("Search Time:", end-start)
    print("Search Results: \n")
    for item in hits:
        print(item.entity.title, " -- ", item.distance,"\n")

   
        
