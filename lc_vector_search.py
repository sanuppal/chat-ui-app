import pinecone
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2') #384 dimensional

#pinecone.init(api_key="89c890aa-ead7-4c56-93d4-57968c3698c2", environment="asia-southeast1-gcp-free")
#index = pinecone.Index("rohit-search-1")

#pinecone.init(api_key="2d1b9675-5d6a-4a31-9a1f-7e1d7095b593", environment="us-west4-gcp-free")
#index = pinecone.Index("penfed-data-index")

##******velux index*********##
pinecone.init(api_key="89c890aa-ead7-4c56-93d4-57968c3698c2", environment="asia-southeast1-gcp-free")
index = pinecone.Index("velux-index-1")

##
# def addData(corpusData,url):
#    id = id = index.describe_index_stats()['total_vector_count']
#    for i in range(len(corpusData)):
#        chunk=corpusData[i]
#        chunkInfo=(str(id+i),
#                model.encode(chunk).tolist(),
#               {'title': url,'context': chunk})
#        index.upsert(vectors=[chunkInfo])
##
def find_match(query):
    docs,res = find_match_private(query,1)
    context= "\n\n".join(res)
    return context

def find_match_private(query,k):
    query_em = model.encode(query).tolist()
    result = index.query(query_em, top_k=k, includeMetadata=True)
    
    return [result['matches'][i]['metadata']['title'] for i in range(k)],[result['matches'][i]['metadata']['context'] for i in range(k)]
