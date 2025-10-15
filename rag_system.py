"""
Local RAG System using Ollama and ChromaDB
Supports loading text files, creating embeddings, and querying with context
"""

import os
import glob
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import requests
import json

class LocalRAG:
    def __init__(self, 
                 knowledge_dir: str = "./knowledge",
                 ollama_url: str = "http://localhost:11434",
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "gpt-oss:20b"):
        """
        Initialize RAG system
        
        Args:
            knowledge_dir: Directory containing .txt files
            ollama_url: Ollama API endpoint
            embedding_model: Model for embeddings (nomic-embed-text recommended)
            llm_model: Model for generation (gpt-oss:20b, etc.)
        """
        self.knowledge_dir = knowledge_dir
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="./chroma_db"
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama"""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text
            }
        )
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f"Embedding failed: {response.text}")
    
    def read_file_with_fallback(self, file_path: str) -> str:
        """Read file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"  ✓ Read with {encoding} encoding")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all fail, read as binary and decode with errors='ignore'
        print(f"  ⚠ Using binary mode with error handling")
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        return content
    
    def load_documents(self):
        """Load all .txt files from knowledge directory"""
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)
            print(f"Created directory: {self.knowledge_dir}")
            print("Please add your .txt files to this directory and run again.")
            return
        
        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {self.knowledge_dir}")
            return
        
        print(f"Found {len(txt_files)} text files. Loading...")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []
        
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")
            
            try:
                content = self.read_file_with_fallback(file_path)
            except Exception as e:
                print(f"  ✗ Failed to read {filename}: {e}")
                continue
            
            chunks = self.chunk_text(content)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filename}_{idx}"
                
                # Get embedding
                embedding = self.get_embedding(chunk)
                
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
                all_metadata.append({
                    "source": filename,
                    "chunk_id": idx
                })
                all_ids.append(chunk_id)
        
        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadata,
            ids=all_ids
        )
        
        print(f"✓ Loaded {len(all_chunks)} chunks from {len(txt_files)} files")
    
    def query(self, question: str, n_results: int = 3) -> str:
        """Query the RAG system"""
        # Get embedding for question
        question_embedding = self.get_embedding(question)
        
        # Search for relevant chunks
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )
        
        # Extract context
        contexts = results['documents'][0]
        sources = [meta['source'] for meta in results['metadatas'][0]]
        
        # Build prompt
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are a helpful assistant. Use the provided context to answer the question accurately.

Context:
{context_str}

Question: {question}

Answer based on the context above. If the information is not in the context, say "I don't have enough information in the provided context to answer that question."

Answer:"""
        
        # Generate response with Ollama
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            }
        )
        
        if response.status_code == 200:
            answer = response.json()["response"]
            return {
                "answer": answer,
                "sources": list(set(sources)),
                "contexts": contexts
            }
        else:
            raise Exception(f"Generation failed: {response.text}")
    
    def chat(self):
        """Interactive chat interface"""
        print("\n=== RAG System Ready ===")
        print("Type your questions (or 'quit' to exit)\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...\n")
            
            try:
                result = self.query(question)
                print(f"Assistant: {result['answer']}\n")
                print(f"Sources: {', '.join(result['sources'])}\n")
            except Exception as e:
                print(f"Error: {e}\n")


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = LocalRAG(
        knowledge_dir="./knowledge",  # Put your .txt files here
        embedding_model="nomic-embed-text",  # or "gpt-oss:20b" if you want to use it for embeddings
        llm_model="gpt-oss:20b"  # Your GPT OSS 20B model
    )
    
    # Load documents
    print("Loading documents...")
    rag.load_documents()
    
    # Start interactive chat
    rag.chat()
