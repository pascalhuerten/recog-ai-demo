import json

from recog_ai import RecognitionAssistant


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
    from recog_ai.utils import extract_json

    raw = 'Here is some text before {"title": "Test"} and after'
    parsed = extract_json(raw)
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
    assistant = RecognitionAssistant(DummyDB(modules))
    suggestions = assistant.get_module_suggestions(
        "doc", institution="Technische Hochschule Lübeck"
    )
    assert len(suggestions) == 1
    assert suggestions[0]["institution"] == "Technische Hochschule Lübeck"


def test_get_modul_info_fallback_on_error():
    class FailingAssistant(RecognitionAssistant):
        def __init__(self, db):
            super().__init__(db)
            # Override LLM client to raise error
            from recog_ai.llm_client import LLMClient

            class FailingClient(LLMClient):
                def invoke(self, messages):
                    raise RuntimeError("chat unavailable")

            self.llm = FailingClient()

    assistant = FailingAssistant(None)
    module = assistant.get_module_info("raw text module")
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

    class StubAssistant(RecognitionAssistant):
        def __init__(self, db):
            super().__init__(db)
            # Override LLM client to return stub response
            from recog_ai.llm_client import LLMClient

            class StubClient(LLMClient):
                def invoke(self, messages):
                    class Result:
                        content = example_response

                    return Result()

            self.llm = StubClient()

    assistant = StubAssistant(None)
    doc = (
        "Das Modul Datenanalyse für Manager bereitet Studierende darauf vor, hochwertige "
        "Reports zu erzeugen. Die Lehrinhalte decken Datenaufbereitung und statistische Methoden ab, "
        "geprüft wird via Klausur innerhalb des Bachelorstudiums."
    )
    module = assistant.get_module_info(doc)
    assert module["title"] == "Datenanalyse für Manager"
    assert module["credits"] == 5
    assert module["learninggoals"] == [
        "Daten aufbereiten",
        "Statistische Auswertungen anwenden",
    ]
    assert module["raw_document"] == doc
