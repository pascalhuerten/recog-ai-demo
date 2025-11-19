import asyncio
import json
import logging
import os
import isodate
import markdown
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

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

logger = logging.getLogger(__name__)


class recognition_assistant:
    def __init__(self, db):
        self.db = db

    def getModuleSuggestions(self, doc, institution=None, limit=5):
        docs = self.db.similarity_search_with_score(doc, limit)

        module_suggestions = []
        for module, score in docs:
            workload = ""
            try:
                workload_iso = module.metadata.get("duration") or module.metadata.get(
                    "workload"
                )
                duration = isodate.parse_duration(workload_iso)
                hours = int(duration.total_seconds() / 3600)
                workload = str(hours) + " Stunden"
            except:
                credits = module.metadata.get("credits")
                if credits:
                    workload = "~" + str(int(credits) * 30) + " Stunden"
                else:
                    workload = ""

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
                "program": self._collect_programs(module.metadata),
                "institution": module_institution,
                "content": module.page_content,
            }
            module_info["json"] = json.dumps(module_info)

            module_suggestions.append(module_info)

        # Return the module suggestions
        return module_suggestions

    def _collect_programs(self, metadata):
        programs = metadata.get("programs") or metadata.get("program")
        if isinstance(programs, (list, tuple)):
            return ", ".join(programs)
        return programs or ""

    def _extract_json(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise

    def _invoke_model(self, model, messages):
        try:
            return model.invoke(messages)
        except ValueError as exc:
            if "Sync client is not available" not in str(exc):
                raise
            logger.info("Sync client unavailable, invoking async model")
            if not hasattr(model, "ainvoke"):
                raise
            return asyncio.run(model.ainvoke(messages))

    def get_chat_model(self, large_model=False, max_tokens=1024):
        thl_chat = ChatOpenAI(
            model=os.getenv("LLM_MODEL"),
            openai_api_base=os.getenv("LLM_URL"),
            openai_api_key=os.getenv("LLM_API_KEY"),
            temperature=0.1,
            max_tokens=max_tokens,
        )

        return thl_chat

    def getModulInfo(self, indoc):
        # Restrict length of doc to 4096 minus the length of the system message

        doc = ""
        try:
            jsondoc = json.loads(doc)
            for key in jsondoc:
                doc += key + ": " + str(jsondoc[key]) + "\n"
        except:
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

        model = self.get_chat_model(False, 4096)
        prompt_value = prompt.invoke(
            {
                "doc": doc,
            }
        )
        messages = prompt_value.to_messages()

        try:
            response = self._invoke_model(model, messages).content
            module = self._extract_json(response)
            if isinstance(module, list):
                module = module[0]

            if (
                module["learninggoals"]
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

    def getExaminationResult(self, module_internal, module_external):
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

        chat = self.get_chat_model(True)

        messages = [
            SystemMessage(content=systemmessage),
            HumanMessage(content=humanmessage),
        ]

        response = chat.invoke(messages).content
        print(response)
        markdown_result = markdown.markdown(response)

        return markdown_result
