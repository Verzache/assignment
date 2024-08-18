# LLM + RAG

To run
1. You need to run llama3.1 on your local machine.
2. In data folder put the PDFs you want your chatbot to know ( context ).
3. run populate_database.py to get the vector embeddings and store it chroma directory.
4. run get_embedding_function.py
5. run query_data.py with your question in this format -> python query_data.py "Your question/Query"
6. run test_rag.py ( Optional ) to see how the model is performing [Unit Testing]

- Note: It is running fully on local machine hence it build with privacy + The OpenAI key you provided is exceeded in limit.
- Additional Improvements: Making a better UI using Streamlit
