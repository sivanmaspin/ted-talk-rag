# TED Talk RAG Assistant 

This project implements a RAG system based on the TED Talks dataset. The system allows users to ask questions and receive answers strictly based on the provided TED transcripts and metadata.

## Architecture
- **Backend:** Flask (Python) deployed on Vercel.
- **Vector Database:** Pinecone.
- **LLM:** Integration with LLMod/OpenAI.

## Live Demo
- **Deployment URL:** [https://ted-talk-rag-beta.vercel.app](https://ted-talk-rag-beta.vercel.app)
- **API Endpoint:** `https://ted-talk-rag-beta.vercel.app/api/prompt`

## API Usage
To interact with the RAG engine, send a **POST** request to the endpoint with the following JSON body:

```json
{
  "question": "What is the dark history of IQ tests?"
}
