from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import requests

# טעינת המפתחות מה-.env
load_dotenv()

app = Flask(__name__)

RAG_CONFIG = {
    "chunk_size": 1000,
    "overlap_ratio": 0.1,
    "top_k": 5
}

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("ted-index")

def get_embedding(text):
    url = "https://api.llmod.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {os.getenv('LLMOD_API_KEY')}"}
    data = {"input": text, "model": "RPRTHPB-text-embedding-3-small"}
    response = requests.post(url, json=data, headers=headers)
    return response.json()['data'][0]['embedding']

@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(RAG_CONFIG)

@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    data = request.json
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    question_embedding = get_embedding(user_question)
    
    results = index.query(
        vector=question_embedding,
        top_k=RAG_CONFIG["top_k"],
        include_metadata=True
    )
    
    context_chunks = []
    context_text_for_ai = ""
    for res in results['matches']:
        metadata = res.get('metadata', {})
        chunk_data = {
            "talk_id": str(metadata.get('talk_id', 'N/A')),
            "title": str(metadata.get('title', 'Unknown Title')),
            "chunk": str(metadata.get('chunk', '')),
            "score": res['score']
        }
        context_chunks.append(chunk_data)
        context_text_for_ai += f"\nTitle: {chunk_data['title']}\nContent: {chunk_data['chunk']}\n"

    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and "
        "only based on the TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not "
        "explicitly contained in the retrieved context. If the answer cannot be determined from "
        "the provided context, respond: “I don’t know based on the provided TED data.” "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant "
        "transcript or metadata when helpful."
    )

    full_user_prompt = f"Context:\n{context_text_for_ai}\n\nQuestion: {user_question}"

    chat_url = "https://api.llmod.ai/v1/chat/completions"
    chat_data = {
        "model": "RPRTHPB-gpt-5-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ]
    }
    
    chat_res = requests.post(
        chat_url, 
        json=chat_data, 
        headers={"Authorization": f"Bearer {os.getenv('LLMOD_API_KEY')}"}
    )
    
    model_answer = chat_res.json()['choices'][0]['message']['content']

    return jsonify({
        "response": model_answer,
        "context": context_chunks,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": full_user_prompt
        }
    })

if __name__ == "__main__":
    app.run(port=3000)