from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
import logging

# Configure logging to be more visible
logging.basicConfig(
    level=logging.DEBUG,  # More detailed logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)
logger = logging.getLogger(__name__)

# Log startup message to verify logging is working
logger.info("=============================================")
logger.info("APPLICATION STARTING")
logger.info("=============================================")

app = Flask(__name__)

# Wrap critical startup functions in try/except blocks
try:
    # Load environment variables
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    logger.info(f"Google API key present: {bool(google_api_key)}")
    
    if not google_api_key:
        logger.critical("GOOGLE_API_KEY not found in environment variables. This will cause failures!")
    
    # Configure Generative AI API - move this up to check API configuration early
    genai.configure(api_key=google_api_key)
    logger.info("Google GenAI API configured")
    
    # Test access to the data files first
    if not os.path.exists("data/extracted_texts_df_ALL.csv"):
        logger.critical("CSV file not found at 'data/extracted_texts_df_ALL.csv'")
        raise FileNotFoundError("CSV file not found")
        
    if not os.path.exists("data/faiss_index.index"):
        logger.critical("FAISS index not found at 'data/faiss_index.index'")
        raise FileNotFoundError("FAISS index not found")

    # Load data and FAISS index
    logger.info("Loading dataframe...")
    df = pd.read_csv("data/extracted_texts_df_ALL.csv")
    logger.info(f"DataFrame loaded successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    
    # Verify DataFrame contains expected columns
    if 'pdf_content' not in df.columns:
        logger.critical("'pdf_content' column not found in DataFrame!")
        raise KeyError("Required column 'pdf_content' not found in DataFrame")
    
    logger.info("Loading FAISS index...")
    index = faiss.read_index("data/faiss_index.index")
    logger.info(f"FAISS index loaded successfully. Dimension: {index.d}, Total vectors: {index.ntotal}")

    # Load the embedder model
    logger.info("Loading sentence transformer model...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    logger.info("Sentence transformer model loaded successfully")

except Exception as e:
    logger.critical(f"CRITICAL ERROR DURING STARTUP: {str(e)}")
    logger.critical(traceback.format_exc())
    raise

# Function to get content from indices with error handling
def get_content_from_indices(indices):
    try:
        logger.info(f"Retrieving content for indices: {indices[0]}")
        contents = []
        for idx in indices[0]:
            if idx < len(df):
                content = df.iloc[idx]['pdf_content']
                # Check if content is valid
                if pd.isna(content) or not content or not isinstance(content, str):
                    logger.warning(f"Empty or invalid content at index {idx}")
                    contents.append("Content appears to be invalid or empty.")
                else:
                    contents.append(content)
                    logger.info(f"Content at index {idx} retrieved successfully (length: {len(content)})")
            else:
                logger.warning(f"Index {idx} out of range for dataframe of length {len(df)}")
                contents.append("Content not found.")
        
        joined_content = "\n\n".join(contents)
        logger.info(f"Total joined content length: {len(joined_content)}")
        return joined_content
    except Exception as e:
        logger.error(f"Error retrieving content: {str(e)}")
        logger.error(traceback.format_exc())
        return "Error retrieving content from database."


# Search function using FAISS and embeddings with error handling
def search(query_text, top_k=1):
    try:
        logger.info(f"Searching for: '{query_text}' with top_k={top_k}")
        
        # Input validation
        if not query_text or not isinstance(query_text, str):
            logger.warning(f"Invalid query text: {query_text}")
            return "No valid query text provided."
        
        # Embed the query
        logger.info("Generating embedding for query")
        query_embedding = embedder.encode(query_text, convert_to_tensor=True)
        logger.info(f"Embedding generated, shape: {query_embedding.shape}")
        query_embedding = query_embedding.cpu().numpy()

        # Normalize the query embedding
        norm = np.linalg.norm(query_embedding)
        logger.info(f"Embedding norm: {norm}")
        if norm == 0:
            logger.warning("Query embedding has zero norm!")
            return "Error: Query embedding could not be normalized."
        
        query_embedding_normalized = query_embedding / norm

        # Reshape to a 2D array for FAISS
        query_embedding_normalized = query_embedding_normalized.reshape(1, -1)
        logger.info(f"Normalized embedding shape: {query_embedding_normalized.shape}")

        # Perform the search
        logger.info("Performing FAISS search")
        distances, indices = index.search(query_embedding_normalized, top_k)
        logger.info(f"Search complete. Found indices: {indices[0]} with distances: {distances[0]}")

        # Get the content based on indices
        content = get_content_from_indices(indices)
        content_preview = content[:100] + "..." if len(content) > 100 else content
        logger.info(f"Retrieved content preview: {content_preview}")
        return content
    
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error during search: {str(e)}"


# Generate the answer using Google's GenAI with robust error handling
def generate_answer(query):
    try:
        logger.info(f"Generating answer for query: '{query}'")
        
        # Input validation
        if not query or not isinstance(query, str):
            logger.warning(f"Invalid query: {query}")
            return "No valid query provided."
        
        # Search for context
        logger.info("Searching for relevant context")
        context = search(query)
        if context.startswith("Error:"):
            logger.error(f"Search failed: {context}")
            return f"Unable to process your query due to a search error: {context}"
        
        logger.info("Context retrieved, preparing prompt")
        
        prompt = f"""
        You are a pharmacy assistant with expertise in providing precise answers based solely on pharmaceutical documents. Your response should adhere to the following guidelines:

        1. **Accuracy**: Ensure that your answer is detailed and 100% accurate, extracted directly from the relevant document content.

        2. **Clarity and Structure**: Organize your response in a clear and reader-friendly format, utilizing headings and bullet points where appropriate for easy understanding.

        3. **Concise Summary**: Include a concise summary at the end for users who prefer a quick answer.

        4. **Handling Unrelated Queries**: If the answer cannot be found in the provided document, clearly state: 'The document does not provide any information for your query' This is important to maintain trust and clarity in communication.

        **Query**:
        "{query}"

        **Context (relevant content from the document)**:
        {context}
        """

        # Initialize the model
        logger.info("Initializing Gemini model")
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Generate content with timeout handling
        logger.info("Sending request to Gemini API")
        try:
            response = model.generate_content(prompt)
            logger.info("Gemini response received successfully")
            
            # Verify response
            if hasattr(response, 'text') and response.text:
                response_preview = response.text[:100] + "..." if len(response.text) > 100 else response.text
                logger.info(f"Response preview: {response_preview}")
                return response.text
            else:
                logger.error("Received empty response from Gemini")
                return "Error: Received empty response from AI model."
                
        except Exception as api_error:
            logger.error(f"Gemini API error: {str(api_error)}")
            return f"Error communicating with AI model: {str(api_error)}"
            
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An unexpected error occurred: {str(e)}"


# Route for the home page with error handling
@app.route('/')
def home():
    try:
        logger.info("Home page requested")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Route for processing the query with error handling
@app.route('/chat', methods=['POST'])
def chat():
    try:
        logger.info("Chat endpoint called")
        
        # Verify request has JSON data
        if not request.is_json:
            logger.error("Request does not contain JSON")
            return jsonify({'error': 'Request must be JSON'}), 400
            
        request_data = request.json
        logger.info(f"Request data: {request_data}")
        
        # Verify query exists in request
        if 'query' not in request_data:
            logger.error("No 'query' field in request")
            return jsonify({'error': 'Missing query field'}), 400
            
        query = request_data['query']
        logger.info(f"Processing query: '{query}'")
        
        response = generate_answer(query)
        logger.info("Response generated, returning to client")
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Add a test endpoint to verify API is working
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'API is running',
        'dataframe_shape': df.shape if 'df' in globals() else None,
        'faiss_index_size': index.ntotal if 'index' in globals() else None,
        'embedder_loaded': embedder is not None if 'embedder' in globals() else None
    })


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True)  # Enable debug mode for more info
    except Exception as e:
        logger.critical(f"Failed to start Flask app: {str(e)}")
        logger.critical(traceback.format_exc())