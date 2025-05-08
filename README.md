# Pharmacy Assistant v2

A smart pharmacovigilance assistant that provides accurate information about medications, side effects, and pharmaceutical queries using advanced natural language processing and semantic search capabilities.

## Data Source

The application's pharmaceutical data has been comprehensively sourced from:
- **Website**: [www.medicines.ie](https://www.medicines.ie/)
- **Data Collection**: All medicines and their associated information have been systematically scraped from the website
- **Dataset Content**: Includes detailed information about:
  - Medication descriptions
  - Side effects
  - Usage instructions
  - Storage requirements
  - Safety information
  - Drug interactions

## Important Note About Data Files

Due to size limitations, the large data files required for this application are hosted separately. Before running the application, you must download these files from:

[Data Files Google Drive Link](https://drive.google.com/drive/folders/1UFqNZLgaONLWUcntWtDQlajgSi4sFNg4?usp=drive_link)

Download and place the following files in a `data` directory in your project root:
- `extracted_texts_df_ALL.csv` (Raw data)
- `faiss_index.index` (Embedded data)

## Application Versions

The repository contains two versions of the application:

### 1. `app.py` (Recommended)
- Uses Google's Gemini-2.0-Flash-Lite model
- Free to use
- Features enhanced error handling and detailed logging
- Comprehensive debugging capabilities
- Good performance for general queries

### 2. `app_2.py` (Alternative)
- Uses OpenAI's GPT model
- Requires paid API access
- Superior performance, especially with GPT-4
- Simpler implementation without extensive logging
- Recommended for professional use where accuracy is critical

## Prerequisites

- Python 3.8+
- pip package manager
- Access to Google Drive (for data files)
- API keys for either Google Gemini AI or OpenAI (depending on version used)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hatimalhomid/pharmacy_assistantv2_489_project.git
cd pharmacy_assistantv2_489_project
Create and activate a virtual environment:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install required packages:
bash
Copy code
pip install -r requirements.txt
Create a .env file and add your API key: For app.py:
Copy code
GOOGLE_API_KEY=your_google_api_key_here
For app_2.py:

Copy code
OPENAI_API_KEY=your_openai_api_key_here
Project Structure
Pharmacy_Assistantv_CSC489_project/
├── app.py                # Main application (Gemini version)
├── app_2.py              # Alternative version (GPT version)
├── requirements.txt      # Project dependencies
│
├── templates/            # Web interface templates
│   └── index.html        # Main HTML template
│
├── static/               # Static assets (CSS, JS)
│   ├── style.css         # Design
│   └── scripts/
│       └── app.js        # For web functionality, STT, and TTS
│
├── data/                 # Data directory for downloaded files
│   ├── extracted_texts_df_ALL.csv
│   └── faiss_index.index
│
└── README.md             # Project documentation

Features
Core Features
Natural language query processing
Semantic search using FAISS
Web-based user interface
Dark/Light theme toggle
Speech-to-text capability
Text-to-speech functionality
Technical Features
Vector similarity search using FAISS
Sentence transformer embeddings
RESTful API endpoints
Comprehensive error handling (in app.py)
Detailed logging system (in app.py)
Running the Application
Ensure all data files are in place and environment variables are set
Choose which version to run:
For Gemini version (Free):

bash
Copy code
python app.py
For GPT version (Paid):

bash
Copy code
python app_2.py
Open a web browser and navigate to http://localhost:8080
Technical Stack
Flask (Web Framework)
FAISS (Vector Similarity Search)
Sentence Transformers (Text Embeddings)
Google Gemini AI / OpenAI GPT (Response Generation)
Pandas (Data Management)
NumPy (Numerical Operations)
Example Usage
The assistant can handle queries such as:

"Does [Medication Name] cause [Adverse Drug Reaction]?"
"What are the side effects of [Medication Name]?"
"How should I store [Medication Name]?"
"Is [Medication Name] safe during pregnancy?"
"What should I do if I miss a dose of [Medication Name]?"
