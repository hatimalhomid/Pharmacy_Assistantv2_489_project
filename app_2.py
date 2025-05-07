from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import openai
import faiss
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load data and FAISS index
df = pd.read_csv("data/extracted_texts_df_ALL.csv")
index = faiss.read_index("data/faiss_index.index")

# Load the embedder model
embedder = SentenceTransformer('all-mpnet-base-v2')

# Function to retrieve content from indices
def get_content_from_indices(indices):
    contents = []
    for idx in indices[0]:
        if idx < len(df):
            contents.append(df.iloc[idx]['pdf_content'])
        else:
            contents.append("Content not found.")
    return "\n\n".join(contents)

# Search function using FAISS and embeddings
def search(query_text, top_k=1):
    # Embed the query
    query_embedding = embedder.encode(query_text, convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()

    # Normalize the query embedding
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

    # Reshape to a 2D array for FAISS
    query_embedding_normalized = query_embedding_normalized.reshape(1, -1)

    # Perform the search
    distances, indices = index.search(query_embedding_normalized, top_k)

    # Get the content based on indices
    content = get_content_from_indices(indices)
    return content

# Load environment variables
load_dotenv()

# Set OpenAI API key

# Generate answer using OpenAI API
def generate_answer(query):
    prompt = f"""
    You are a pharmacy assistant with expertise in providing precise answers based solely on pharmaceutical documents. Your response should adhere to the following guidelines:

    Accuracy: Ensure that your answer is detailed and 100% accurate, extracted directly from the relevant document content.

    Clarity and Structure: Organize your response in a clear and reader-friendly format, utilizing headings and bullet points where appropriate for easy understanding.

    Concise Summary: Include a concise summary at the end for users who prefer a quick answer.

    Handling Unrelated Queries: If the answer cannot be found in the provided document, clearly state: 'The document does not provide any information for your query' This is important to maintain trust and clarity in communication.

    Query:
    "{query}"

    Context (relevant content from the document):
    {search(query)}
    """

    messages = [
        {
            "role": "system",
            "content": "You are a pharmacy assistant providing answers based solely on pharmaceutical documents."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        max_tokens=1500,
        n=1,
        stop=None,
        temperature=0.2,
        messages=messages
    )

    generated_text = response.choices[0].message['content'].strip()
    return generated_text

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for processing the query
@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    response = generate_answer(query)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)