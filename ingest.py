import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
import requests

# טעינת מפתחות
load_dotenv()
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# אתחול Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ted-index")

def get_embedding(text):
    """שליחת טקסט לקבלת וקטור מה-API"""
    url = "https://api.llmod.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {LLMOD_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text, 
        "model": "RPRTHPB-text-embedding-3-small"
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    else:
        raise Exception(f"Error in Embedding: {response.text}")

def chunk_text(text, size=1000):
    """חלוקת הטקסט לקטעים קטנים"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def run_ingestion():
    # קריאת הקובץ המלא
    df = pd.read_csv('ted_talks_en.csv')
    
    # --- שלב הניקוי: מחיקת נתונים קיימים כדי למנוע כפילויות מאתמול ---
    print("Cleaning existing data from Pinecone for a fresh start...")
    index.delete(delete_all=True)
    
    # הגדרת הדגימה לכל הקובץ
    df_sample = df
    total_talks = len(df_sample)
    
    vectors_to_upsert = []
    
    print(f"Starting ingestion of {total_talks} talks...")

    for idx, row in df_sample.iterrows():
        # הדפסת התקדמות (למשל: Processing talk 1/4005)
        print(f"Processing talk {idx+1}/{total_talks}: {row['title']}")
        
        text_chunks = chunk_text(str(row['transcript']))
        
        for i, chunk in enumerate(text_chunks):
            try:
                embedding = get_embedding(chunk)           
                vector_id = f"talk_{row['talk_id']}_{i}"
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "talk_id": str(row['talk_id']),
                        "title": str(row['title']),
                        "chunk": chunk
                    }
                })
            except Exception as e:
                print(f"Error on talk {row['talk_id']}, chunk {i}: {e}")
                continue

            # העלאה ל-Pinecone בקבוצות של 50 כדי לשמור על יציבות
            if len(vectors_to_upsert) >= 50:
                print(f"--- Uploading batch of {len(vectors_to_upsert)} chunks to Pinecone ---")
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = [] 

    # העלאת השאריות שנותרו בבאץ' האחרון
    if vectors_to_upsert:
        print(f"--- Uploading final batch of {len(vectors_to_upsert)} chunks ---")
        index.upsert(vectors=vectors_to_upsert)
    
    print(f"Successfully uploaded all {total_talks} talks to Pinecone!")

if __name__ == "__main__":
    run_ingestion()