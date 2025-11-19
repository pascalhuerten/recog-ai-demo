# Project Title: recog-ai-demo

## Project Description

`recog-ai-demo` is a web application designed to showcase an AI-powered recognition workflow. Leveraging advanced machine learning models (LLM), this application harmonizes and compares various modules on a semantic level.

### Key Features

1. **Module Description Harmonization:**
   - Parses and harmonizes uploaded module descriptions, enhancing comparability with internally stored modules in a vector database.

2. **Internal Module Suggestions:**
   - Utilizes the vector database and semantic similarity to suggest internal modules that are likely to have a high chance of recognition for the uploaded external module.

3. **Module Comparison and Recognition Possibility:**
   - Compares an external module with an internal module, evaluating the possibility of recognition based on predefined criteria.

## Project Structure

The refactored codebase follows a modular, SOLID-aligned architecture:

```
recog_ai/                          # Core recognition package
├── __init__.py                   # Package initialization and exports
├── config.py                     # Configuration and initialization helpers
├── llm_client.py                 # LLM client wrapper with async fallback
├── assistant.py                  # RecognitionAssistant orchestration class
└── utils.py                      # Utility functions (JSON parsing, metadata extraction)

app.py                            # Flask application with cleaned routes
```

### Module Overview

- **`config.py`**: Handles environment loading, embedding initialization, and database setup.
- **`llm_client.py`**: Encapsulates ChatOpenAI client with fallback to async invocation if sync unavailable.
- **`assistant.py`**: `RecognitionAssistant` class orchestrates module parsing, semantic search, and module comparison.
- **`utils.py`**: Reusable utility functions for JSON extraction, workload parsing, and program collection.
- **`app.py`**: Simplified Flask routes leveraging the modular helpers.

## Installation

To install and run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/pascalhuerten/recog-ai-demo.git
   cd recog-ai-demo
   ```

2. Create a `.env` file in the project root and set your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here or any openai compatible api key
   LLM_URL=API_endpoint_of_openai_compatible_api
   LLM_MODEL=gemma-3-27b-it # or any model supported by your API
   HOST_PORT=80 # or any port you prefer
   ```

3. Create a vector store (or use pre-existing):

   - Prepare your module descriptions in a suitable format (e.g., JSON, plain text).
   - Modify the application code to read the module descriptions and create a vector store using chromadb. Update the code adjusting paths and configurations as needed.

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Build the Docker image:

   ```bash
   docker build -t recog-ai-demo .
   ```

6. Run the Docker container:

   ```bash
   docker run -p 80:80 recog-ai-demo
   ```

The application will be accessible at `http://localhost:80`.

## Development

### Running Tests

Unit tests and integration tests are provided:

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/test_recog_ai.py -v

# Run integration tests (requires LLM credentials)
pytest tests/integration/ -v

# Run tests with coverage report
pytest tests/ --cov=recog_ai --cov=app
```

### Running Locally

For development without Docker:

```bash
# Install dependencies in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the Flask development server
python app.py
```

The app will be available at `http://localhost:5000`.

## How to Use

1. Access the application at `http://localhost:80`.
2. Upload a module description file (PDF, TXT, or XML) or enter the description in the provided text area.
3. Click on "Find Modules" to get module suggestions based on the description.
4. Select an external module and an internal module for comparison.
5. Click on "Select Module" to see the examination result, including recognition possibility.
