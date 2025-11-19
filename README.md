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

#### Installation

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

3. Creating a Vector Store (Alternative to proprietary vector store):

   - Prepare your module descriptions in a suitable format (e.g., JSON, plain text).
   - Modify the application code to read the module descriptions and create a vector store using chromadb. Update the code adjusting paths and configurations as needed.

4. Build the Docker image:

   ```bash
   docker build -t recog-ai-demo .
   ```

5. Run the Docker container:

   ```bash
   docker run -p 80:80 recog-ai-demo
   ```

The application will be accessible at `http://localhost:80`.

#### How to Use

1. Access the application at `http://localhost:80`.
2. Upload a module description file (PDF, TXT, or XML) or enter the description in the provided text area.
3. Click on "Find Modules" to get module suggestions based on the description.
4. Select an external module and an internal module for comparison.
5. Click on "Select Module" to see the examination result, including recognition possibility.