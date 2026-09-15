"""Corpus loading tests."""

import pytest

from retrieveval.ingestion.loaders import CorpusError, load_corpus


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "billing.md").write_text("Billing content", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Plain text content", encoding="utf-8")
    (tmp_path / "faq").mkdir()
    (tmp_path / "faq" / "cancellation.md").write_text("How to cancel", encoding="utf-8")
    return tmp_path


class TestLoading:
    def test_reads_supported_files_recursively(self, corpus):
        documents = load_corpus(corpus)
        assert [d.id for d in documents] == ["billing.md", "faq/cancellation.md", "notes.txt"]

    def test_document_ids_are_corpus_relative_posix_paths(self, corpus):
        nested = next(d for d in load_corpus(corpus) if d.id.startswith("faq"))
        assert nested.id == "faq/cancellation.md"
        assert nested.text == "How to cancel"

    def test_keeps_the_source_path(self, corpus):
        document = next(d for d in load_corpus(corpus) if d.id == "billing.md")
        assert document.source_path == str(corpus / "billing.md")

    def test_ordering_is_stable(self, corpus):
        assert [d.id for d in load_corpus(corpus)] == [d.id for d in load_corpus(corpus)]

    def test_ignores_hidden_files_and_directories(self, corpus):
        (corpus / ".DS_Store").write_text("junk", encoding="utf-8")
        (corpus / ".git").mkdir()
        (corpus / ".git" / "config.md").write_text("not corpus", encoding="utf-8")
        assert len(load_corpus(corpus)) == 3


class TestPdf:
    def test_extracts_text_from_a_pdf(self, tmp_path, make_pdf):
        (tmp_path / "refunds.pdf").write_bytes(
            make_pdf("Annual plans are refunded within 14 days")
        )
        document = load_corpus(tmp_path)[0]
        assert document.id == "refunds.pdf"
        assert "refunded within 14 days" in document.text

    def test_a_pdf_sits_alongside_text_documents(self, corpus, make_pdf):
        (corpus / "refunds.pdf").write_bytes(make_pdf("Refund policy"))
        assert len(load_corpus(corpus)) == 4

    def test_a_pdf_with_no_text_layer_is_reported_not_silently_indexed(
        self, tmp_path, make_pdf
    ):
        # This is the scanned-document case: a valid PDF whose pages carry no
        # text. Indexing it as an empty document would show up much later as an
        # unexplained recall ceiling.
        (tmp_path / "scan.pdf").write_bytes(make_pdf(""))
        with pytest.raises(CorpusError, match="OCR"):
            load_corpus(tmp_path)

    def test_a_corrupt_pdf_names_the_file(self, tmp_path):
        (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\nnot really a pdf")
        with pytest.raises(CorpusError, match="broken.pdf"):
            load_corpus(tmp_path)


class TestDocx:
    def test_extracts_paragraphs(self, tmp_path, make_docx):
        make_docx(
            tmp_path / "policy.docx",
            paragraphs=["Refund Policy", "Annual plans are refunded within 14 days."],
        )
        document = load_corpus(tmp_path)[0]
        assert document.id == "policy.docx"
        assert "Refund Policy" in document.text
        assert "refunded within 14 days" in document.text

    def test_extracts_table_content(self, tmp_path, make_docx):
        # Tables in a Word document routinely hold real content, not layout.
        # Dropping them would silently lose it.
        make_docx(
            tmp_path / "plans.docx",
            paragraphs=["Plan comparison"],
            table=[["Plan", "Refund window"], ["Annual", "14 days"]],
        )
        text = load_corpus(tmp_path)[0].text
        assert "Refund window" in text
        assert "Annual" in text and "14 days" in text

    def test_keeps_paragraphs_and_tables_in_document_order(self, tmp_path, make_docx):
        make_docx(
            tmp_path / "ordered.docx",
            paragraphs=["FIRST paragraph"],
            table=[["MIDDLE cell"]],
        )
        text = load_corpus(tmp_path)[0].text
        assert text.index("FIRST") < text.index("MIDDLE")

    def test_sits_alongside_other_formats(self, corpus, make_docx):
        make_docx(corpus / "policy.docx", paragraphs=["Refund policy"])
        assert len(load_corpus(corpus)) == 4

    def test_a_docx_with_no_text_is_reported(self, tmp_path, make_docx):
        make_docx(tmp_path / "blank.docx", paragraphs=[])
        with pytest.raises(CorpusError, match="no text could be extracted"):
            load_corpus(tmp_path)

    def test_a_corrupt_docx_names_the_file(self, tmp_path):
        (tmp_path / "broken.docx").write_bytes(b"not a zip archive at all")
        with pytest.raises(CorpusError, match="broken.docx"):
            load_corpus(tmp_path)

    def test_a_renamed_legacy_doc_gets_a_usable_hint(self, tmp_path):
        # Renaming .doc to .docx is a common and confusing mistake: the file
        # opens fine in Word but is a completely different binary format.
        (tmp_path / "legacy.docx").write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
        with pytest.raises(CorpusError, match=r"Legacy `\.doc`"):
            load_corpus(tmp_path)


class TestHtml:
    def test_extracts_prose(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<html><body><h1>Exports</h1>"
            "<p>Exports are delivered as a ZIP archive.</p></body></html>",
            encoding="utf-8",
        )
        text = load_corpus(tmp_path)[0].text
        assert "Exports" in text
        assert "ZIP archive" in text

    def test_ignores_script_and_style_content(self, tmp_path):
        # A wiki export carries analytics and CSS that would otherwise be
        # indexed as if it were documentation.
        (tmp_path / "page.html").write_text(
            "<html><head><style>.x{color:red}</style>"
            "<script>var tracking = 'do_not_index_me';</script></head>"
            "<body><p>Real content here.</p></body></html>",
            encoding="utf-8",
        )
        text = load_corpus(tmp_path)[0].text
        assert "Real content here." in text
        assert "do_not_index_me" not in text
        assert "color:red" not in text

    def test_resolves_html_entities(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<p>Billing &amp; invoices &mdash; see the FAQ.</p>", encoding="utf-8"
        )
        text = load_corpus(tmp_path)[0].text
        assert "Billing & invoices" in text
        assert "&amp;" not in text

    def test_separates_block_elements_so_chunking_can_split_on_them(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<p>First paragraph.</p><p>Second paragraph.</p>", encoding="utf-8"
        )
        text = load_corpus(tmp_path)[0].text
        assert "First paragraph.\n\nSecond paragraph." in text

    def test_does_not_run_words_together_across_tags(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<li>alpha</li><li>beta</li>", encoding="utf-8"
        )
        assert "alphabeta" not in load_corpus(tmp_path)[0].text

    def test_does_not_index_the_head_title(self, tmp_path):
        # Confluence and Notion exports set <title> to the same string as the
        # <h1>. Counting both would double every page's heading and inflate
        # lexical scores for HTML documents purely because of their format.
        (tmp_path / "page.html").write_text(
            "<html><head><title>API Keys</title></head>"
            "<body><h1>API Keys</h1><p>Keys can be revoked.</p></body></html>",
            encoding="utf-8",
        )
        text = load_corpus(tmp_path)[0].text
        assert text.count("API Keys") == 1

    def test_ignores_other_head_metadata(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<html><head><title>T</title>"
            "<meta name='description' content='ignored'></head>"
            "<body><p>Body content.</p></body></html>",
            encoding="utf-8",
        )
        assert load_corpus(tmp_path)[0].text == "Body content."

    def test_recovers_when_the_head_is_never_closed(self, tmp_path):
        # Without a </head>, treating <body> as the boundary is what keeps the
        # entire document from being skipped.
        (tmp_path / "page.html").write_text(
            "<html><head><title>Ignored</title>"
            "<body><p>Body survives.</p></body></html>",
            encoding="utf-8",
        )
        text = load_corpus(tmp_path)[0].text
        assert "Body survives." in text
        assert "Ignored" not in text

    def test_a_page_with_no_head_still_works(self, tmp_path):
        (tmp_path / "page.html").write_text(
            "<p>Fragment with no head or body tags.</p>", encoding="utf-8"
        )
        assert "Fragment with no head" in load_corpus(tmp_path)[0].text

    def test_accepts_the_htm_extension(self, tmp_path):
        (tmp_path / "page.htm").write_text("<p>Legacy extension.</p>", encoding="utf-8")
        assert load_corpus(tmp_path)[0].id == "page.htm"

    def test_tolerates_malformed_markup(self, tmp_path):
        # Exported HTML is frequently not well-formed; recovering the prose
        # matters more than rejecting the file.
        (tmp_path / "page.html").write_text(
            "<p>Unclosed paragraph <b>bold text <p>Another one.", encoding="utf-8"
        )
        text = load_corpus(tmp_path)[0].text
        assert "Unclosed paragraph" in text
        assert "Another one." in text

    def test_an_html_file_with_no_prose_is_reported(self, tmp_path):
        (tmp_path / "empty.html").write_text(
            "<html><head><script>var x = 1;</script></head><body></body></html>",
            encoding="utf-8",
        )
        with pytest.raises(CorpusError, match="no text could be extracted"):
            load_corpus(tmp_path)

    def test_the_ocr_hint_is_not_shown_for_non_pdf_files(self, tmp_path):
        (tmp_path / "empty.html").write_text("<body></body>", encoding="utf-8")
        with pytest.raises(CorpusError) as caught:
            load_corpus(tmp_path)
        assert "OCR" not in str(caught.value)


class TestErrors:
    def test_missing_directory(self, tmp_path):
        with pytest.raises(CorpusError, match="corpus directory not found"):
            load_corpus(tmp_path / "nope")

    def test_path_is_a_file(self, corpus):
        with pytest.raises(CorpusError, match="not a directory"):
            load_corpus(corpus / "billing.md")

    def test_unsupported_file_types_are_reported_by_name(self, corpus):
        (corpus / "budget.xlsx").write_bytes(b"binary")
        with pytest.raises(CorpusError, match="budget.xlsx"):
            load_corpus(corpus)

    def test_unsupported_file_types_can_be_skipped(self, corpus):
        (corpus / "budget.xlsx").write_bytes(b"binary")
        assert len(load_corpus(corpus, skip_unsupported=True)) == 3

    def test_empty_corpus(self, tmp_path):
        with pytest.raises(CorpusError, match="no supported documents"):
            load_corpus(tmp_path)

    def test_a_file_with_no_extractable_text_is_reported(self, corpus):
        (corpus / "blank.md").write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(CorpusError, match="no text could be extracted"):
            load_corpus(corpus)

    def test_the_ocr_limitation_is_stated_when_a_pdf_yields_no_text(
        self, corpus, make_pdf
    ):
        (corpus / "scan.pdf").write_bytes(make_pdf(""))
        with pytest.raises(CorpusError, match="OCR"):
            load_corpus(corpus)

    def test_non_utf8_text_is_reported(self, corpus):
        (corpus / "latin1.txt").write_bytes("café".encode("latin-1"))
        with pytest.raises(CorpusError, match="not valid UTF-8"):
            load_corpus(corpus)


class TestDocxTableRegressions:
    def test_a_row_may_repeat_a_value_in_adjacent_columns(self, tmp_path, make_docx):
        # Comparing cell *text* to collapse merged cells also collapsed
        # legitimate repeats, so "Export | Yes | Yes" lost a column and the row
        # no longer lined up with its header.
        make_docx(
            tmp_path / "matrix.docx",
            paragraphs=["Feature matrix"],
            table=[["Feature", "Standard", "Enterprise"], ["Export", "Yes", "Yes"]],
        )
        text = load_corpus(tmp_path)[0].text
        assert "Export | Yes | Yes" in text

    def test_a_horizontally_merged_cell_is_not_repeated(self, tmp_path):
        import docx

        document = docx.Document()
        table = document.add_table(rows=1, cols=3)
        merged = table.rows[0].cells[0].merge(table.rows[0].cells[1])
        merged.text = "Spans two columns"
        table.rows[0].cells[2].text = "Third"
        document.save(str(tmp_path / "merged.docx"))

        text = load_corpus(tmp_path)[0].text
        assert text.count("Spans two columns") == 1
        assert "Third" in text
