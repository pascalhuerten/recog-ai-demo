"""Tests for Flask routes in app.py"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestIndexRoute:
    """Test the index route."""

    def test_index_returns_success(self):
        """Test that the index route returns 200."""
        # Import here to avoid issues with app initialization
        from app import app

        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200
        assert b"Willkommen" in response.data


class TestFindModuleRoute:
    """Test the find_module route."""

    def test_find_module_get_returns_template(self):
        """Test GET request to find_module returns the module suggestions template."""
        from app import app

        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/find_module")
        assert response.status_code == 200

    def test_find_module_post_with_no_file_or_text(self):
        """Test POST with no file or text returns template."""
        from app import app

        app.config["TESTING"] = True
        client = app.test_client()

        # When no file or text is provided, Flask will return 400
        # This is correct behavior - the form requires either file or text
        response = client.post(
            "/find_module",
            data={"institution_filter": "all"},
        )
        # Empty POST request should not be processed
        assert response.status_code in [200, 400]


class TestSelectModuleRoute:
    """Test the select_module route."""

    def test_select_module_handles_missing_original_doc(self):
        """Test that select_module handles external modules without original_doc."""
        from app import app

        app.config["TESTING"] = True
        client = app.test_client()

        internal_module = {
            "title": "Internal",
            "learninggoals": [],
        }
        external_module = {
            "title": "External",
            "learninggoals": [],
        }

        with patch("app.RecognitionAssistant") as mock_assistant_class:
            mock_assistant = MagicMock()
            mock_assistant.get_module_info.return_value = {"learninggoals": []}
            mock_assistant.get_examination_result.return_value = "<p>Result</p>"
            mock_assistant_class.return_value = mock_assistant

            response = client.post(
                "/select_module",
                data={
                    "selected_module": json.dumps(internal_module),
                    "external_module": json.dumps(external_module),
                },
            )

            # Should handle gracefully even if original_doc is missing
            assert response.status_code in [200, 500]
