import os
from dotenv import load_dotenv
from pinecone import Pinecone

# טעינת המפתחות מהקובץ .env
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "ted-index"

try:
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print("Successfully connected to Pinecone!")
    print("Index Stats:", stats)
except Exception as e:
    print("Connection failed. Check your .env file.")
    print(f"Error: {e}")