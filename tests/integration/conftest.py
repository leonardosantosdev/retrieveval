"""A tiny corpus and dataset shared by the integration tests.

Small enough to reason about by hand, but built so that a lexical and a
semantic retriever do not behave identically on it: some queries repeat the
corpus vocabulary, others paraphrase it.
"""

import json

import pytest

DOCUMENTS = {
    "cancellation.md": (
        "# Cancelling\n\n"
        "You can end your subscription from the account area at any time. "
        "Access continues until the close of the current billing period.\n\n"
        "Your workspace is kept for 90 days after the plan ends, and is then "
        "permanently removed."
    ),
    "billing.md": (
        "# Billing\n\n"
        "Invoices are issued on the first day of each billing period and charged "
        "to the payment method on file.\n\n"
        "A declined charge is retried after 3 days and again after 7 days."
    ),
    "authentication.md": (
        "# Authentication\n\n"
        "The API authenticates requests with a bearer token.\n\n"
        "An access token is valid for 60 minutes. After that the API responds "
        "with HTTP 401 and the error code TOKEN_EXPIRED."
    ),
    "notes.txt": "Internal note: the staging workspace is reset every Sunday night.",
}

QUERIES = [
    {"query": "How do I cancel my subscription?", "relevant_documents": ["cancellation.md"]},
    {"query": "TOKEN_EXPIRED", "relevant_documents": ["authentication.md"]},
    {"query": "when do you retry a declined card", "relevant_documents": ["billing.md"]},
    {"query": "when is staging wiped", "relevant_documents": ["notes.txt"]},
]


@pytest.fixture
def corpus_dir(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()
    for name, text in DOCUMENTS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def dataset_file(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(QUERIES, indent=2), encoding="utf-8")
    return path
