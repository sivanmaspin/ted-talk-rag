from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import requests

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
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TED Talk RAG Assistant</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f8f9fa; color: #333; }
            .card { background: white; padding: 2.5rem; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center; max-width: 450px; }
            h1 { color: #e62b1e; margin-bottom: 0.5rem; font-size: 2rem; }
            p { color: #666; line-height: 1.6; }
            .status-container { margin: 1.5rem 0; padding: 0.75rem; background: #e7f5ea; border-radius: 8px; }
            .status { color: #1e7e34; font-weight: 600; font-size: 0.9rem; }
            code { background: #eee; padding: 0.2rem 0.4rem; border-radius: 4px; font-family: monospace; }
            hr { border: 0; border-top: 1px solid #eee; margin: 1.5rem 0; }
            .footer { font-size: 0.8rem; color: #999; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>TED Talk RAG 🎤</h1>
            <p>Your Intelligent TED Assistant is officially live and running in the cloud.</p>
            <div class="status-container">
                <span class="status">● Systems Operational: Pinecone & LLMod Connected</span>
            </div>
            <hr>
            <p>To interact with the RAG engine, send a <strong>POST</strong> request to:</p>
            <code>/api/prompt</code>
            <div style="margin-top: 2rem;" class="footer">
                Developed as part of the Information Engineering Project
            </div>
        </div>
    </body>
    </html>
    """
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