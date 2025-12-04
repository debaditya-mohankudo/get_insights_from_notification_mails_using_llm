import pytest
from tag_classifier import generate_tags_from_pr_title, classify_tags


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Fix crash when DB table missing", ["bug", "sql"]),
        ("Database query optimisation", ["performance", "sql"]),
        ("Improve UI layout for dashboard", ["ui"]),
        ("Update API endpoint to return JSON", ["api"]),
        ("Security patch for XSS issue", ["bug", "security"]),
        ("Speed up response latency", ["performance"]),
        ("Refactor code" , []),
        ("", []),
        (None, []),
    ],
)
def test_generate_tags_from_pr_title(title, expected):
    assert generate_tags_from_pr_title(title) == expected


def test_classify_tags_combines_rules():
    title = "Fix API error and optimize DB query in UI"
    tags = classify_tags(title)

    # Expecting multiple categories to match
    assert set(tags) == {"api", "bug", "performance", "sql", "ui"}


def test_no_false_positives():
    title = "Add documentation and README updates"
    tags = classify_tags(title)
    assert tags == []


def test_case_insensitivity():
    title = "FIX Bug IN UI MODULE"
    tags = classify_tags(title)
    assert set(tags) == {"bug", "ui"}


def test_partial_word_should_not_match():
    title = "rebuild tableView component"  # tableView should NOT trigger 'table'
    tags = classify_tags(title)
    assert "sql" not in tags
