import json
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage, SystemMessage
import markdown
import os
import isodate
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class ModuleSchema(BaseModel):
    title: str = Field(description="Titel des Moduls")
    credits: Optional[int] = Field(None, description="ECTS-Punkte des Moduls")
    workload: Optional[str] = Field(
        None,
        description="Arbeitsaufwand des Moduls in Stunden pro Semester. Beispiel: 'X Stunden'",
    )
    learninggoals: List[str] = Field(
        description="Lernziele des Moduls. Jedes Lernziel ist ein einfacher String. Beispiel: ['Lernziel 1', 'Lernziel 2']"
    )
    assessmenttype: Optional[str] = Field(None, description="Prüfungsform des Moduls")
    level: Optional[str] = Field(
        None, description="Bildungsniveau des Moduls. 'Bachelor' oder 'Master'"
    )


class recognition_assistant:
    def __init__(self, db):
        self.db = db

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
            model="gemma-3-27b-it",
            openai_api_base=os.getenv("LARGE_CUSTOM_LLM_URL"),
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
        
        template = (
            "Folgendes Dokument ist gegeben:\n"
            "{doc}\n\n"
            "Analysiere die gegebene Modulbeschreibung oder erbrachte Leistung und extrahiere die relevanten Metadaten.\n\n"
            "{format_instructions}\n\n"
            "Gebe nun die extrahierten Modulinformation in deutscher Sprache und konform zum angegeben Schema aus.\n"
        )

        parser = PydanticOutputParser(pydantic_object=ModuleSchema)
        model = self.get_chat_model(False, 4096)
        prompt = PromptTemplate(
            template=template,
            input_variables=["doc"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | model | parser

        try:
            module = chain.invoke({"doc": doc})
            # to dict
            module = module.dict()
            # if module is a list, take the first element
            if isinstance(module, list):
                module = module[0]

            if module["learninggoals"] and len(module["learninggoals"]) > 0 and isinstance(module["learninggoals"][0], dict):
                strlist = []
                # Get values of the dict in the list
                for item in module["learninggoals"]:
                    for key, value in item.items():
                        strlist.append(value)
                module["learninggoals"] = strlist
            
            print(module)
            # Remove all characters before and after {}
            module["original_doc"] = doc
        except Exception as e:
            print(e)
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
