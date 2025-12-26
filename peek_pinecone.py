import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ted-index")

# ננסה לשלוף וקטור אחד כדי לראות את המבנה שלו
results = index.query(
    vector=[0]*1536, # וקטור ריק רק כדי לקבל תוצאות
    top_k=1,
    include_metadata=True
)

if results['matches']:
    print("Found a vector! Here is the metadata:")
    print(results['matches'][0]['metadata'])
else:
    print("Index is empty or no metadata found.")