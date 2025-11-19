import json

from recog_ai import recognition_assistant


class DummyModule:
    def __init__(self, metadata, content=""):
        self.metadata = metadata
        self.page_content = content


class DummyDB:
    def __init__(self, modules):
        self.modules = modules

    def similarity_search_with_score(self, doc, limit):
        return [(module, idx) for idx, module in enumerate(self.modules[:limit])]


def test_extract_json_with_surrounding_text():
    assistant = recognition_assistant(None)
    raw = 'Here is some text before {"title": "Test"} and after'
    parsed = assistant._extract_json(raw)
    assert parsed["title"] == "Test"


def test_module_suggestions_filters_institution():
    modules = [
        DummyModule(
            {
                "title": "A",
                "credits": 5,
                "description": "desc",
                "level": "Bachelor",
                "programs": ["Program"],
                "institution": "Technische Hochschule Lübeck",
                "duration": "P0Y1M0DT0H0M0S",
            }
        ),
        DummyModule(
            {
                "title": "B",
                "credits": 4,
                "description": "desc",
                "level": "Master",
                "programs": ["Other"],
                "institution": "Universität Bielefeld",
                "duration": "P0Y1M0DT0H0M0S",
            }
        ),
    ]
    assistant = recognition_assistant(DummyDB(modules))
    suggestions = assistant.getModuleSuggestions(
        "doc", institution="Technische Hochschule Lübeck"
    )
    assert len(suggestions) == 1
    assert suggestions[0]["institution"] == "Technische Hochschule Lübeck"


def test_get_modul_info_fallback_on_error():
    class FailingAssistant(recognition_assistant):
        def get_chat_model(self, large_model=False, max_tokens=1024):
            class Model:
                def invoke(self_inner, messages):
                    raise RuntimeError("chat unavailable")

            return Model()

    assistant = FailingAssistant(None)
    module = assistant.getModulInfo("raw text module")
    assert module["raw_document"] == "raw text module"
    assert module["description"] == "raw text module"
    assert module["error"] == "chat unavailable"


def test_get_modul_info_parses_german_text():
    example_response = json.dumps(
        {
            "title": "Datenanalyse für Manager",
            "credits": 5,
            "workload": "90 Stunden",
            "learninggoals": [
                "Daten aufbereiten",
                "Statistische Auswertungen anwenden",
            ],
            "assessmenttype": "Klausur",
            "level": "Bachelor",
            "description": "Analytisches Denken und Reporting",
            "program": "Wirtschaftswissenschaften",
            "institution": "Technische Hochschule Lübeck",
        }
    )

    class StubAssistant(recognition_assistant):
        def get_chat_model(self, large_model=False, max_tokens=1024):
            class Model:
                def invoke(self_inner, messages):
                    class Result:
                        content = example_response

                    return Result()

            return Model()

    assistant = StubAssistant(None)
    doc = (
        "Das Modul Datenanalyse für Manager bereitet Studierende darauf vor, hochwertige "
        "Reports zu erzeugen. Die Lehrinhalte decken Datenaufbereitung und statistische Methoden ab, "
        "geprüft wird via Klausur innerhalb des Bachelorstudiums."
    )
    module = assistant.getModulInfo(doc)
    assert module["title"] == "Datenanalyse für Manager"
    assert module["credits"] == 5
    assert module["learninggoals"] == [
        "Daten aufbereiten",
        "Statistische Auswertungen anwenden",
    ]
    assert module["raw_document"] == doc
