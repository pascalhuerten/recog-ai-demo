from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from recog_ai import recognition_assistant
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma
import os
import openai
from PyPDF2 import PdfReader
import json
from dotenv import load_dotenv

project_folder = os.path.expanduser('~/wisyki-api')
load_dotenv(os.path.join(project_folder, '.env'))

app = Flask(__name__)
CORS(app)


@app.before_first_request
def load_instructor():
    global instructor
    instructor = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        embed_instruction="Represent the document for retrieval: ",
        query_instruction="Represent the query for retrieval: "
    )
    dir = os.path.dirname(__file__)
    persist_directory = dir + '/data/thl_modules_vectorstore'
    chroma_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    
    global moduledb
    moduledb = Chroma(persist_directory=persist_directory, embedding_function=instructor, client_settings=chroma_settings)
    
    global recognition_ai
    recognition_ai = recognition_assistant(moduledb)
        


@app.route("/", methods=['GET'])
def index():
    return jsonify({"about": "This is a prototype for an ai-assisted recognition workflow."})


# Endpunkt für die Startseite
@app.route('/find_module', methods=['GET', 'POST'])
def find_module():
    if request.method == 'POST':
        # Hier verarbeiten wir den Dateiupload und rufen getModuleSuggestions() auf.
        doc = None
        uploaded_file = request.files['file']
        if uploaded_file:
            # Check if it's a PDF file
            if uploaded_file.filename.endswith('.pdf'):
                pdf = PdfReader(uploaded_file)
                doc = ""
                for page_num in range(len(pdf.pages)):
                    page = pdf.pages[page_num]
                    doc += page.extract_text()
            # Check if it's a TXT file
            elif uploaded_file.filename.endswith('.txt'):
                doc = uploaded_file.read().decode('utf-8')
            # Check if it's a xml file
            elif uploaded_file.filename.endswith('.xml'):
                doc = uploaded_file.read().decode('utf-8')
            else:
                raise Exception('File type not supported')
        else:
            doc = request.form['text']

        print(doc[:50])

        if not doc:
            return render_template('module_suggestions.html')
        
        module_suggestions = recognition_ai.getModuleSuggestions(doc)
        external_module_json = recognition_ai.getModulInfo(doc)
        print(external_module_json)
        external_module_parsed = json.loads(external_module_json)

        return render_template('module_suggestions.html', module_suggestions=module_suggestions, external_module_parsed=external_module_parsed, external_module_json=external_module_json)
        
    return render_template('module_suggestions.html')


# Endpunkt für die Modulauswahl und Prüfung
@app.route('/select_module', methods=['POST'])
def select_module():
    internal_module_json = request.form['selected_module']
    external_module_json = request.form['external_module']

    print(internal_module_json)
    print(external_module_json)

    external_module_parsed  = json.loads(external_module_json)

    internal_module_json = recognition_ai.getModulInfo(internal_module_json)
    internal_module_parsed = json.loads(internal_module_json)
    
    # Hier rufen wir getExaminationResult() auf und generieren das Prüfungsergebnis.
    examination_result = recognition_ai.getExaminationResult(internal_module_json, external_module_json)
    
    return render_template('examination_result.html', internal_module_parsed=internal_module_parsed, external_module_parsed=external_module_parsed, examination_result=examination_result)


if __name__ == '__main__':
    app.run(debug=False)
