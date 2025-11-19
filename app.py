from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
import pdfplumber
import json
import os
from dotenv import load_dotenv
from recog_ai import recognition_assistant
from visualize.visualize import visualize_bp, initChromaviz

INSTITUTION_FILTERS = [
    {"value": "all", "label": "Alle Hochschulen"},
    {"value": "Technische Hochschule Lübeck", "label": "Technische Hochschule Lübeck"},
    {"value": "Universität Bielefeld", "label": "Universität Bielefeld"},
]


app = Flask(__name__)
app.register_blueprint(visualize_bp)
CORS(app)


def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="isy-thl/multilingual-e5-base-course-skill-tuned",
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
    )


def load_moduledb(embedding):
    return Chroma(
        client=chromadb.PersistentClient(
            os.path.dirname(__file__) + "/data/modules_vectorstore"
        ),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


embedding = load_embedding()
moduledb = load_moduledb(embedding)

initChromaviz(moduledb._collection)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Endpunkt für die Startseite
@app.route("/find_module", methods=["GET", "POST"])
def find_module():
    institution_filter = "all"
    if request.method == "POST":
        # Hier verarbeiten wir den Dateiupload und rufen getModuleSuggestions() auf.
        institution_filter = request.form.get("institution_filter", "all")
        doc = None
        uploaded_file = request.files["file"]
        if uploaded_file:
            # Check if it's a PDF file
            if uploaded_file.filename.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    doc = ""
                    for page_num in range(max(2, len(pdf.pages))):
                        doc += pdf.pages[page_num].extract_text()
            # Check if it's a TXT file
            elif uploaded_file.filename.endswith(".txt"):
                doc = uploaded_file.read().decode("utf-8")
            # Check if it's a xml file
            elif uploaded_file.filename.endswith(".xml"):
                doc = uploaded_file.read().decode("utf-8")
            else:
                raise Exception("File type not supported")
        else:
            doc = request.form["text"]

        if not doc:
            return render_template("module_suggestions.html")

        # No more than 10000 characters
        doc = doc[:10000]

        recog_assistant = recognition_assistant(moduledb)

        external_module_parsed = recog_assistant.getModulInfo(doc)
        external_module_json = json.dumps(external_module_parsed)
        translated_doc = ""
        if external_module_parsed["title"]:
            translated_doc += "Titel: \n"
            translated_doc += external_module_parsed["title"]
            translated_doc += "\n"
        if external_module_parsed["learninggoals"]:
            translated_doc += "Lernziele: \n"
            translated_doc += "\n".join(external_module_parsed["learninggoals"])
            translated_doc += "\n"
        if external_module_parsed["level"]:
            translated_doc += "Niveau: \n"
            translated_doc += external_module_parsed["level"]
            translated_doc += "\n"
        if not translated_doc.strip():
            translated_doc = external_module_parsed.get("raw_document", doc)[:10000]
        module_suggestions = recog_assistant.getModuleSuggestions(
            translated_doc, institution=institution_filter
        )

        return render_template(
            "module_suggestions.html",
            module_suggestions=module_suggestions,
            external_module_parsed=external_module_parsed,
            external_module_json=external_module_json,
            institution_filter=institution_filter,
            institution_filters=INSTITUTION_FILTERS,
        )

    return render_template(
        "module_suggestions.html",
        institution_filter=institution_filter,
        institution_filters=INSTITUTION_FILTERS,
    )


# Endpunkt für die Modulauswahl und Prüfung
@app.route("/select_module", methods=["POST"])
def select_module():
    recog_assistant = recognition_assistant(moduledb)
    internal_module_json = request.form["selected_module"]
    internal_module_parsed = json.loads(internal_module_json)

    # Get learninggoals
    internal_module_ai_parsed = recog_assistant.getModulInfo(internal_module_json)
    internal_module_parsed["learninggoals"] = internal_module_ai_parsed["learninggoals"]

    external_module_json = request.form["external_module"]
    external_module_parsed = json.loads(external_module_json)

    # Original_doc is not needed for processing of the examination result
    tmp = json.loads(external_module_json)
    del tmp["original_doc"]
    external_module_json = json.dumps(tmp)

    # Hier rufen wir getExaminationResult() auf und generieren das Prüfungsergebnis.
    examination_result = recog_assistant.getExaminationResult(
        internal_module_json, external_module_json
    )

    return render_template(
        "examination_result.html",
        internal_module_parsed=internal_module_parsed,
        external_module_parsed=external_module_parsed,
        examination_result=examination_result,
    )


if __name__ == "__main__":
    app.run(debug=True)
