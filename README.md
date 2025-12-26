# TED Talk RAG Assistant

This project implements a RAG system based on the TED Talks dataset. Answers are strictly based on provided transcripts and metadata.

## Architecture
- **Backend:** Flask (Python) on Vercel.
- **Vector Database:** Pinecone.
- **LLM:** RPRTHPB-gpt-5-mini.

## Configuration
- **Chunk Size:** 1000.
- **Overlap Ratio:** 0.1.
- **Top-k:** 5.

## Live Demo
- **Deployment URL:** https://ted-talk-rag-beta.vercel.app
- **Query Endpoint (POST):** /api/prompt
- **Stats Endpoint (GET):** /api/stats

## API Usage
Post a question to `/api/prompt`:
```json
{
  "question": "Give me a list of 3 talk titles about education."
}
