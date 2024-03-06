import json
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import markdown
import os
import isodate


class recognition_assistant:
    def __init__(self, db):
        self.db = db
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def getModuleSuggestions(self, doc):
        docs = self.db.similarity_search_with_score(doc, 5)

        module_suggestions = []
        for module, score in docs:
            workload = ""
            try:
                workload_iso = module.metadata["workload"]
                duration = isodate.parse_duration(workload_iso)
                hours = int(duration.total_seconds() / 3600)
                workload = str(hours) + " Stunden"
            except:
                workload = "~" + str(int(module.metadata["credits"]) * 30) + " Stunden"

            module_info = {
                "title": module.metadata["title"],
                "credits": module.metadata["credits"],
                "workload": workload,
                "description": module.metadata["description"],
                "level": module.metadata["level"],
                "program": module.metadata["program"],
                "content": module.page_content,
            }
            module_info["json"] = json.dumps(module_info)

            module_suggestions.append(module_info)

        # Return the module suggestions
        return module_suggestions

    def get_chat_model(self, large_model=False, max_tokens=1024):
        thl_chat = ChatOpenAI(
            model="mixtral-8x7b",
            openai_api_base="https://mixtral-8x7b.llm.mylab.th-luebeck.dev/v1",
            openai_api_key="-",
            temperature=0.1,
            max_tokens=max_tokens,
        )

        mistral_chat = ChatMistralAI(
            model="mistral-mdeium" if not large_model else "mistral-large",
            temperature=0.1,
            max_tokens=max_tokens,
        )

        return thl_chat.with_fallbacks([mistral_chat])

    def getModulInfo(self, doc):
        # Restrict length of doc to 4096 minus the length of the system message
        doc = doc[: 4096 - 512]
        messages = [
            SystemMessage(
                content=(
                    "Folgendes Doument ist gegeben:"
                    ""
                    "Analysiere die gegebenen Modulbeschreibungen und fülle anschließend die Lücken in folgendem JSON sinvoll und in Deutscher Sprache aus. "
                    "Der Workload sollte in Stunden pro Semester angegeben sein. Das Level bezieht sich auf das Bildungsniveau des Kurses und kann nur \"Bachelor\" oder \"Master\" enthalten. "
                    "Wenn eine passende Information in der gegebenen Modulbeschreibung fehlt, soll das Attribut den Wert null bekommen. "
                    "Achte darauf, dass das Ergebnis valides JSON ist und befolge das folgende JSON Schema genau."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "Folgende Modulbeschreibung ist gegeben:"
                "{doc}"
                ""
                "Der JSON OUTPUT beschreibt genau ein Modul als JSON Objekt und die Inhalte sind sauber ins deutsche übersetzt."
                "Die Antwort enthält keine weiteren Informationen als das eine JSON Objekt."
                ""
                "JSON Schema:"
                "{schema}"
                ""
                "Ausgefülltes JSON:"
            )
        ]

        chat = self.get_chat_model(False, 512)
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | chat | StrOutputParser()

        schema ="""
{
    "title": "",
    "credits": "",
    "workload": " Stunden",
    "learninggoals": [],
    "assessmenttype": "",
    "level": ""
}
"""

        try:
            response = chain.invoke({"doc": doc, "schema": schema})
            print(response)
            # Remove all characters before and after {}
            response = response[response.find("{") : response.rfind("}") + 1]
            module = json.loads(response)
            module["original_doc"] = doc
        except Exception as e:
            raise e
            module = {
                "title": "",
                "credits": "",
                "workload": "",
                "learninggoals": [],
                "assessmenttype": "",
                "level": "",
            }

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
Gib an dieser Stelle zusätzlich den Hinweis, dass das Ergebnis auf Basis eines generativen Sprachmodelles namens mixtral-8x7b generiert wurde. Das Open-Source Modell wurde von MistralAI entwickelt und wird von der Technischen Hochschule Lübeck bereitgestellt.
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

        chat  = self.get_chat_model(True)

        messages = [
            SystemMessage(content=systemmessage),
            HumanMessage(content=humanmessage),
        ]
    	
        response = chat.invoke(messages).content
        print(response)
        markdown_result = markdown.markdown(response)

        return markdown_result
