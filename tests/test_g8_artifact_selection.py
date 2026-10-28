"""Fail-closed G8 artifact selection request and redirect safety."""
import unittest

from fukaji.g8_artifact_selection import (
    allowed_artifact_redirect, expected_request, validate_request,
)


class G8ArtifactSelectionProtocolTests(unittest.TestCase):
    def test_request_is_exact_and_initially_disabled(self):
        self.assertFalse(validate_request(expected_request(False)))
        self.assertTrue(validate_request(expected_request(True)))
        for change in (
            {"source_run_id": 0},
            {"source_artifact_sha256": "0" * 64},
            {"old_artifact_id": 0},
            {"training_allowed": True},
            {"adoption_allowed": True},
            {"selection_seed": "other"},
            {"unexpected": "extra"},
        ):
            cfg = expected_request(True)
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(cfg)

    def test_signed_redirect_host_restriction(self):
        good = (
            "https://productionresultssa0.blob.core.windows.net/a?sig=abc",
            "https://pipelines.actions.githubusercontent.com/a",
            "https://objects.githubusercontent.com/a",
        )
        for url in good:
            self.assertTrue(allowed_artifact_redirect(url), url)
        bad = (
            "http://productionresultssa0.blob.core.windows.net/a",
            "https://example.com/a",
            "https://blob.core.windows.net/a",
            "https://productionresultssa0.blob.core.windows.net.evil.com/a",
            "https://user:pass@objects.githubusercontent.com/a",
            "https://api.github.com/a",
        )
        for url in bad:
            self.assertFalse(allowed_artifact_redirect(url), url)


if __name__ == "__main__":
    unittest.main()
