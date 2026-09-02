"""Tests for the small contract helpers in governance/db/models.py."""

from governance.db.models import TEST_RESULT_STATUSES, is_pass


def test_is_pass_is_true_only_for_an_explicit_pass():
    assert is_pass("pass") is True
    # everything else is NOT a pass — a lazy `status != "fail"` would get these wrong
    for other in ("warn", "fail", "indeterminate", None, "", "PASS", "passed"):
        assert is_pass(other) is False


def test_indeterminate_is_a_recognised_status():
    assert "indeterminate" in TEST_RESULT_STATUSES
    assert set(TEST_RESULT_STATUSES) == {"pass", "warn", "fail", "indeterminate"}
