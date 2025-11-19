import os
import pytest
from dotenv import load_dotenv

from recog_ai import recognition_assistant

load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.getenv("LLM_MODEL") and os.getenv("LLM_API_KEY") and os.getenv("LLM_URL")),
    reason="integration test requires LLM credentials",
)


def test_get_modul_info_against_llm():
    """Hit the configured LLM and verify that a simple German description is parsed."""

    assistant = recognition_assistant(None)
    doc = (
        "Das Modul Führungskompetenz vermittelt Managementmethoden und bereitet Studierende "
        "darauf vor, klare Entscheidungen treffen zu können. Die Leistung wird durch eine schriftliche "
        "Klausur nach Abschluss des Bachelor-Moduls geprüft."
    )
    module = assistant.getModulInfo(doc)
    assert module["title"], "LLM should provide a title because it has enough context"
    assert module["learninggoals"], "LLM should extract at least one learning goal"
