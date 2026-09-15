"""The command line interface.

Two commands: ``evaluate`` runs a benchmark, ``index`` only builds and caches
the corpus embeddings. ``index`` exists because embedding a large corpus is the
slow part and is worth doing once, deliberately, rather than discovering it in
the middle of a benchmark run.

Every expected failure -- a missing corpus, a malformed dataset, a label that
does not match any document, an unavailable model -- is reported as a single
clear line and a non-zero exit code. Stack traces are reserved for bugs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from .config import Config, ConfigError
from .evaluation.dataset import DatasetError, load_dataset, validate_against_corpus
from .evaluation.runner import metric_names, run_benchmark
from .ingestion.chunking import chunk_documents
from .ingestion.loaders import CorpusError, load_corpus
from .reporting.report import build_run, print_report, write_artifacts
from .retrieval.dense import ModelUnavailableError
from .retrieval.factory import build_strategies

def _quieten_model_downloads() -> None:
    """Stop Hugging Face progress bars and hub notices from interleaving with
    the report. The presentation layer owns the terminal, so the suppression
    belongs here rather than inside the retrievers."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark retrieval strategies on your own corpus.",
)

console = Console()
errors = Console(stderr=True)

EXPECTED_FAILURES = (ConfigError, CorpusError, DatasetError, ModelUnavailableError)

CorpusOption = Annotated[
    Path, typer.Option("--corpus", "-c", help="Directory containing the documents.")
]
ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="YAML configuration file. Defaults apply if omitted.")
]
SkipUnsupportedOption = Annotated[
    bool,
    typer.Option("--skip-unsupported", help="Ignore files whose type cannot be read."),
]


def _fail(message: str) -> NoReturn:
    errors.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _load_config(path: Path | None) -> Config:
    return Config.load(path) if path is not None else Config()


def _step(message: str) -> None:
    console.print(f"[dim]{message}...[/dim]")


@app.command()
def evaluate(
    corpus: CorpusOption,
    dataset: Annotated[
        Path, typer.Option("--dataset", "-d", help="JSON file of labeled evaluation queries.")
    ],
    config_path: ConfigOption = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory for the JSON results artifact.")
    ] = Path("results"),
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write the JSON results artifact.")
    ] = True,
    skip_unsupported: SkipUnsupportedOption = False,
) -> None:
    """Benchmark every configured retrieval strategy against a labeled dataset."""
    _quieten_model_downloads()
    try:
        configuration = _load_config(config_path)

        _step(f"Reading corpus from {corpus}")
        documents = load_corpus(corpus, skip_unsupported=skip_unsupported)

        chunks = chunk_documents(documents, configuration.chunking)
        console.print(
            f"[dim]{len(documents)} documents -> {len(chunks)} chunks "
            f"(size {configuration.chunking.size}, overlap {configuration.chunking.overlap})[/dim]"
        )

        _step(f"Reading evaluation dataset from {dataset}")
        queries = load_dataset(dataset)
        validate_against_corpus(
            queries,
            (document.id for document in documents),
            (chunk.id for chunk in chunks),
        )
        console.print(f"[dim]{len(queries)} labeled queries[/dim]")

        strategies = build_strategies(chunks, configuration, on_step=_step)

        _step(f"Running {len(queries)} queries against {len(strategies)} strategies")
        results = run_benchmark(strategies, queries, configuration.evaluation)
    except EXPECTED_FAILURES as exc:
        _fail(str(exc))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _fail("interrupted")

    keys = metric_names(configuration.evaluation)
    headline_metric = f"ndcg@{configuration.evaluation.max_k}"
    print_report(console, results, keys, headline_metric)

    if save:
        run = build_run(
            results,
            configuration,
            corpus={
                "path": str(corpus),
                "documents": len(documents),
                "chunks": len(chunks),
            },
            dataset={"path": str(dataset), "queries": len(queries)},
        )
        json_path, markdown_path = write_artifacts(run, output, keys, headline_metric)
        console.print(f"\n[dim]Results written to {json_path} and {markdown_path}[/dim]")


@app.command()
def index(
    corpus: CorpusOption,
    config_path: ConfigOption = None,
    skip_unsupported: SkipUnsupportedOption = False,
) -> None:
    """Build and cache the corpus embeddings, so a later evaluate run is fast."""
    _quieten_model_downloads()
    try:
        configuration = _load_config(config_path)
        if not configuration.needs_dense_model:
            console.print(
                "[dim]No dense or hybrid retriever is configured, so there is nothing "
                "to cache.[/dim]"
            )
            raise typer.Exit(code=0)

        _step(f"Reading corpus from {corpus}")
        documents = load_corpus(corpus, skip_unsupported=skip_unsupported)
        chunks = chunk_documents(documents, configuration.chunking)
        console.print(f"[dim]{len(documents)} documents -> {len(chunks)} chunks[/dim]")

        from .retrieval.dense import DenseRetriever

        _step(f"Embedding corpus with {configuration.dense.model}")
        retriever = DenseRetriever(
            configuration.dense.model,
            batch_size=configuration.dense.batch_size,
            cache_dir=configuration.dense.cache_dir,
        )
        retriever.index(chunks)
    except EXPECTED_FAILURES as exc:
        _fail(str(exc))

    console.print(
        f"[green]Embeddings ready[/green] for {len(chunks)} chunks "
        f"in {configuration.dense.cache_dir}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
