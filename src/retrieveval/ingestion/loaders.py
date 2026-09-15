"""Reading a corpus directory into :class:`~retrieveval.models.Document` objects.

Document identity is the corpus-relative path with forward slashes
(``faq/billing.md``), because that is what a user naturally writes in their
evaluation dataset. Nothing downstream invents or rewrites ids.

This layer is strict on purpose. A corpus file that silently fails to load
becomes a document that can never be retrieved, which shows up much later as an
unexplained recall ceiling. Better to fail at load time and name the file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from html.parser import HTMLParser
from pathlib import Path

from ..models import Document

TEXT_EXTENSIONS = frozenset({".txt", ".md"})
PDF_EXTENSIONS = frozenset({".pdf"})
DOCX_EXTENSIONS = frozenset({".docx"})
HTML_EXTENSIONS = frozenset({".html", ".htm"})

__all__ = ["CorpusError", "SUPPORTED_EXTENSIONS", "load_corpus"]


class CorpusError(Exception):
    """Raised when a corpus cannot be read as specified."""


def _document_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_hidden(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorpusError(
            f"{path} is not valid UTF-8 text ({exc.reason}); "
            "convert it to UTF-8 or remove it from the corpus"
        ) from exc


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CorpusError("reading PDFs requires the `pypdf` package") from exc

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise CorpusError(f"failed to read PDF {path}: {exc}") from exc
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    """Extract a Word document's prose, in document order.

    ``python-docx`` exposes body paragraphs and tables as separate collections,
    which loses their interleaving. Walking the body element instead keeps a
    table's rows where the author put them, so chunks stay coherent.

    Headers, footers and footnotes are skipped: they are usually page furniture
    repeated on every page, and indexing them would have every chunk of a long
    document share the same boilerplate.
    """
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise CorpusError(
            f"reading {path.suffix} files requires the `python-docx` package; "
            'install it with `pip install "retrieveval[office]"`'
        ) from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise CorpusError(
            f"failed to read Word document {path}: {exc}. "
            "Legacy `.doc` files renamed to `.docx` are a common cause; "
            "re-save the file as a real .docx."
        ) from exc

    blocks: list[str] = []
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(element, document).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            for row in Table(element, document).rows:
                # A horizontally merged cell is reported once per column it
                # spans, always as the same underlying XML element. Identity is
                # what distinguishes that from two separate cells that happen
                # to hold the same value -- comparing text would collapse a
                # legitimate "Export | Yes | Yes" into "Export | Yes" and
                # silently misalign the row against its header.
                values: list[str] = []
                previous = None
                for cell in row.cells:
                    if cell._tc is previous:
                        continue
                    previous = cell._tc
                    text = cell.text.strip()
                    if text:
                        values.append(text)
                if values:
                    blocks.append(" | ".join(values))
    return "\n\n".join(blocks)


class _HtmlTextExtractor(HTMLParser):
    """Collects the readable text of an HTML document.

    Uses the standard library rather than a parsing dependency: the job here is
    only to recover prose from exported wiki pages, not to model the DOM.
    """

    #: Elements whose content is code or styling, never prose.
    SKIPPED = frozenset({"script", "style", "noscript", "template", "svg"})

    #: Elements that end a line of prose. Marking these keeps paragraphs apart,
    #: which matters because the chunker splits on blank lines first.
    BLOCKS = frozenset({
        "p", "div", "br", "hr", "li", "ul", "ol", "dl", "dt", "dd",
        "h1", "h2", "h3", "h4", "h5", "h6", "table", "tr", "td", "th",
        "section", "article", "header", "footer", "aside", "nav",
        "blockquote", "pre", "figure", "figcaption", "main", "form",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_head = False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "head":
            # <title> is browser chrome, not page prose, and exported wiki
            # pages (Confluence, Notion) set it to the same string as the <h1>.
            # Collecting both would count every page's heading twice and
            # inflate its term frequency, biasing lexical retrieval towards
            # HTML documents purely because of their format.
            self._in_head = True
        elif tag == "body":
            # Exported HTML frequently omits </head>; reaching <body> is the
            # reliable signal that the head is over.
            self._in_head = False
        elif tag in self.SKIPPED:
            self._skip_depth += 1
        elif tag in self.BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._in_head = False
        elif tag in self.SKIPPED:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and not self._in_head:
            self._parts.append(data)

    def text(self) -> str:
        collected = "".join(self._parts)
        collected = re.sub(r"[ \t\r\f\v]+", " ", collected)
        collected = "\n".join(line.strip() for line in collected.splitlines())
        # A block start and a block end each emit a newline, so adjacent
        # paragraphs already separate by two; anything more is just markup depth.
        return re.sub(r"\n{3,}", "\n\n", collected).strip()


def _read_html(path: Path) -> str:
    parser = _HtmlTextExtractor()
    try:
        parser.feed(_read_text_file(path))
        parser.close()
    except CorpusError:
        raise
    except Exception as exc:
        raise CorpusError(f"failed to read HTML document {path}: {exc}") from exc
    return parser.text()


#: Extension -> extractor. A registry rather than a chain of conditionals,
#: because the set of formats is now the thing most likely to grow.
_EXTRACTORS: dict[str, Callable[[Path], str]] = {
    **{extension: _read_text_file for extension in TEXT_EXTENSIONS},
    **{extension: _read_pdf for extension in PDF_EXTENSIONS},
    **{extension: _read_docx for extension in DOCX_EXTENSIONS},
    **{extension: _read_html for extension in HTML_EXTENSIONS},
}

SUPPORTED_EXTENSIONS = frozenset(_EXTRACTORS)


def _extract(path: Path) -> str:
    return _EXTRACTORS[path.suffix.lower()](path)


def _describe(paths: Iterable[Path], root: Path, limit: int = 10) -> str:
    names = sorted(_document_id(path, root) for path in paths)
    shown = ", ".join(names[:limit])
    remaining = len(names) - limit
    return f"{shown} (and {remaining} more)" if remaining > 0 else shown


def load_corpus(root: str | Path, *, skip_unsupported: bool = False) -> list[Document]:
    """Load every supported document under ``root``, recursively.

    Hidden files and directories (any path component starting with ``.``) are
    always ignored. Unsupported file types raise unless ``skip_unsupported``
    is set, so that a corpus of PDFs plus a stray ``.docx`` is not quietly
    benchmarked as if the ``.docx`` did not exist.
    """
    corpus_root = Path(root)
    if not corpus_root.exists():
        raise CorpusError(f"corpus directory not found: {corpus_root}")
    if not corpus_root.is_dir():
        raise CorpusError(f"corpus path is not a directory: {corpus_root}")

    supported: list[Path] = []
    unsupported: list[Path] = []
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file() or _is_hidden(path, corpus_root):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            unsupported.append(path)

    if unsupported and not skip_unsupported:
        message = (
            f"unsupported file type(s) in {corpus_root}: "
            f"{_describe(unsupported, corpus_root)}. "
            f"Supported extensions are {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
        )
        if any(path.suffix.lower() == ".doc" for path in unsupported):
            # Common enough in an established corpus to be worth naming.
            message += (
                "Legacy `.doc` is a different, binary format and is not supported; "
                "re-save those files as `.docx`. "
            )
        raise CorpusError(message + "Pass --skip-unsupported to ignore them.")
    if not supported:
        raise CorpusError(
            f"no supported documents found in {corpus_root} "
            f"(looked for {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    documents: list[Document] = []
    empty: list[Path] = []
    for path in supported:
        text = _extract(path)
        if not text.strip():
            empty.append(path)
            continue
        documents.append(
            Document(id=_document_id(path, corpus_root), text=text, source_path=str(path))
        )

    if empty:
        message = f"no text could be extracted from: {_describe(empty, corpus_root)}. "
        if any(path.suffix.lower() in PDF_EXTENSIONS for path in empty):
            message += (
                "Scanned or image-only PDFs need OCR, which RetrievEval does not perform. "
            )
        raise CorpusError(
            message + "Remove these files or replace them with a text version."
        )
    return documents
