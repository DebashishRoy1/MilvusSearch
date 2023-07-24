from pymilvus import Collection, utility,connections
from milvus import default_server
import pandas as pd
import csv
connections.connect(host='127.0.0.1', port=default_server.listen_port)
collection = Collection("DiabetesStudy")
collection.load()
utility.load_state("DiabetesStudy")
utility.loading_progress("DiabetesStudy")
#print(collection.num_entities)
TOPK = 20
def embed_search(data):
    return [data]

df = pd.read_csv("./Prevent/DiabetesStudy.csv")
cleaned_df = df.dropna()

StoreResult = []
i=1
for index,row in cleaned_df.iterrows():
    patientId=row["Patient_ID"]
    searchData= row.drop("Patient_ID").to_numpy()
    search_data = embed_search(searchData)
    res= collection.search(
        data=search_data,
        anns_field ="embedding",
        param = {"metric_type": "L2", "params":{"nprobe":10}},
        limit=TOPK,
        output_fields = ["Patient_ID"]
        )
    
    #print("Patients similar to: ",patientId )
    for hits_i, hits in enumerate(res):
        for item in hits:
            listRow=[]
            listRow.append(patientId)
            listRow.append(item.entity.Patient_ID)
            listRow.append(item.distance)
            StoreResult.append(listRow)


f = open('similarityScore.csv', 'w',newline='')
writer = csv.writer(f)

for row in StoreResult:
    writer.writerow(row)

f.close()
    
    
