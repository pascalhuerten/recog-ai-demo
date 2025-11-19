"""Recognition assistant for module extraction and comparison."""

import json
import logging
import markdown
from typing import List, Dict, Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from recog_ai.llm_client import LLMClient
from recog_ai.utils import extract_json, parse_workload, collect_programs

logger = logging.getLogger(__name__)

MODULE_SCHEMA = {
    "title": {"type": "string", "description": "Titel des Moduls"},
    "credits": {"type": "number", "minimum": 0, "description": "ECTS-Punkte"},
    "workload": {"type": ["string", "null"], "description": "Arbeitsaufwand"},
    "learninggoals": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Lernziele",
    },
    "assessmenttype": {"type": ["string", "null"], "description": "Prüfungsform"},
    "level": {"type": ["string", "null"], "description": "Bildungsniveau"},
    "program": {"type": ["string", "null"], "description": "Studiengänge"},
    "institution": {"type": ["string", "null"], "description": "Institution"},
}


class RecognitionAssistant:
    """Orchestrates module parsing, suggestion, and recognition workflows."""

    def __init__(self, moduledb: Any, llm_client: Optional[LLMClient] = None) -> None:
        """
        Initialize the recognition assistant.

        Args:
            moduledb: Chroma vector database instance for similarity search.
            llm_client: Optional LLMClient instance; if None, creates a default one.
        """
        self.db = moduledb
        self.llm = llm_client or LLMClient()

    def get_module_suggestions(
        self, doc: str, institution: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve module suggestions based on semantic similarity.

        Args:
            doc: Input document/query string.
            institution: Optional institution filter.
            limit: Maximum number of suggestions to return.

        Returns:
            List of module suggestion dictionaries.
        """
        docs = self.db.similarity_search_with_score(doc, limit)

        module_suggestions = []
        for module, score in docs:
            workload = parse_workload(module.metadata)
            module_institution = module.metadata.get("institution")

            if institution and module_institution:
                if (
                    module_institution.strip().lower() != institution.strip().lower()
                    and institution.lower() != "all"
                ):
                    continue

            module_info = {
                "title": module.metadata.get("title")
                or module.metadata.get("name")
                or "",
                "credits": module.metadata.get("credits"),
                "workload": workload,
                "description": module.metadata.get("description")
                or module.metadata.get("learning_outcomes")
                or "",
                "level": module.metadata.get("level"),
                "program": collect_programs(module.metadata),
                "institution": module_institution,
                "content": module.page_content,
            }
            module_info["json"] = json.dumps(module_info)
            module_suggestions.append(module_info)

        return module_suggestions

    def get_module_info(self, indoc: str) -> Dict[str, Any]:
        """
        Extract structured module metadata from unstructured text using LLM.

        Falls back to raw text if extraction fails.

        Args:
            indoc: Raw module document/description text.

        Returns:
            Dictionary with extracted fields (title, credits, learninggoals, etc.)
            or fallback structure with raw_document and error fields.
        """
        doc = ""
        try:
            jsondoc = json.loads(indoc)
            for key in jsondoc:
                doc += key + ": " + str(jsondoc[key]) + "\n"
        except Exception:
            doc = indoc

        schema_json = json.dumps(MODULE_SCHEMA, indent=2, sort_keys=True)
        schema_json_safe = (
            schema_json.replace("{", "{{").replace("}", "}}").replace("\n", "\n")
        )
        system_prompt = (
            "Du bekommst eine akademische Modulbeschreibung. Extrahiere alle relevanten Metadaten und gib sie als JSON-Objekt aus.\n\n"
            "Die Antwort muss ausschließlich gültiges JSON sein, das mit dem folgenden Schema übereinstimmt:\n"
            f"{schema_json_safe}\n"
            "Nutze deutsche Feldbeschreibungen und vermeide zusätzlichen Fließtext.\n"
            "Wenn du Informationen nicht hast, verwende leere Strings oder leere Listen."
        )
        human_prompt = (
            "Folgendes Dokument ist gegeben:\n"
            "{doc}\n\n"
            "Achte auf Titel, Credits, Lernziele, Bildungsniveau, Arbeitsaufwand und Prüfungsform."
        )

        prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

        llm = LLMClient(max_tokens=4096)
        prompt_value = prompt.invoke({"doc": doc})
        messages = prompt_value.to_messages()

        try:
            response = self.llm.invoke(messages).content
            module = extract_json(response)
            if isinstance(module, list):
                module = module[0]

            if (
                module.get("learninggoals")
                and len(module["learninggoals"]) > 0
                and isinstance(module["learninggoals"][0], dict)
            ):
                strlist = []
                for item in module["learninggoals"]:
                    for _, value in item.items():
                        strlist.append(value)
                module["learninggoals"] = strlist

            logger.info("Extracted module info with title=%s", module.get("title"))
            module["original_doc"] = doc
            module["raw_document"] = doc
        except Exception as e:
            logger.exception("Module extraction failed, falling back to raw text")
            module = {
                "title": "",
                "credits": "",
                "workload": "",
                "learninggoals": [],
                "assessmenttype": "",
                "level": "",
                "description": doc,
                "program": "",
                "institution": "",
            }
            module["original_doc"] = doc
            module["raw_document"] = doc
            module["error"] = str(e)

        return module

    def get_examination_result(self, module_internal: str, module_external: str) -> str:
        """
        Compare two modules and generate an HTML-formatted examination result.

        Args:
            module_internal: JSON string of the internal module.
            module_external: JSON string of the external module.

        Returns:
            HTML-formatted examination result as a string.
        """
        systemmessage = """
Ich bin als KI-Assistent*in im Prüfungsamt einer Hochschule tätig. Meine Hauptaufgaben umfassen die Beantwortung von Fragen zu Modulen und die Überprüfung, ob ein externes Modul auf ein internes Modul anerkannt werden kann.

Folgende Kriterien werden bei der Prüfung der Anerkennbarkeit berücksichtigt:
- Lernziele
- ECTS-Punkte/Credits
- Arbeitsaufwand
- Bildungsniveau
- Prüfungsform

Bei der Bewertung werden diese Kriterien gleichwertig berücksichtigt.
Eine Ausnahme ist der Arbeitsaufwand. Dieser sollte nicht in die Bewertung einfließen, wenn die Infromationen dazu nicht gut vergleichbar sind.
Beide Module sollten möglichst demselben Bildungsniveau (Bachelor oder Master) entsprechen.
Wenn das externe Modul mehr Credits aufweist als das interne Modul oder die Diskrepanz etwa 10 Prozent beträgt, ist dies kein Grund für eine Nichtanerkennung. Wenn das interne Modul jedoch signifikant mehr Credits hat als das externe Modul, kann höchstens eine teilweise Anerkennung erfolgen.
Das Kriterium der Prüfungsform sollte nicht berücksichtigt werden wenn, diese Informatione nicht für beide Module vorliegt und nicht vergleichbar ist.

Es gibt drei mögliche Ergebnisse für die Prüfung:
- Vollständige Anerkennung, wenn mindestens 80 Prozent der Lernziele übereinstimmen
- Teilweise Anerkennung, wenn mindestens 50 Prozent der Lernziele übereinstimmen
- Keine Anerkennung, wenn nur wenige oder keine Lernziele übereinstimmen

Die Abschnitte und Inhalte meiner Antworten strukturiere ich mit Markdown. Kriterien werden einzeln bewertet. Lernziele müssen nur bei Unterschieden aufgelistet werden.
Am Schluss der Prüfung folgt eine prägnante, hervorgehobene Zusammenfassung des Prüfungsergebnisses mit dem Ergebnis: "Es wird auf Basis des Vergelichs der Module  eine *Vollständige Anerkennung*, *Teilweise Anerkennung* oder *Keine Anerkennung* empfohlen.
Gib an dieser Stelle zusätzlich den Hinweis, dass das Ergebnis auf Basis eines generativen OpenSource-Sprachmodelles namens gemma-3-27b-it generiert wurde. Das Open-Source Modell wird von [KISSKI](https://kisski.gwdg.de) bereitgestellt.
        """

        humanmessage = (
            """
## Externes Modul

            """
            + module_external
            + """


## Internes Modul:

            """
            + module_internal
            + """
        """
        )

        llm = LLMClient(max_tokens=2048)
        messages = [
            SystemMessage(content=systemmessage),
            HumanMessage(content=humanmessage),
        ]

        response = self.llm.invoke(messages).content
        logger.info("Generated examination result")
        markdown_result = markdown.markdown(response)

        return markdown_result
