"""Rendering benchmark results.

Two outputs, deliberately different in kind: a console table for reading now,
and a JSON artifact that records the configuration alongside the numbers so a
result can be reproduced or compared against a later run.

The summary names the leaders on specific, stated metrics. It does not rank
strategies overall or recommend one. Which trade-off is right depends on
things this tool cannot see -- latency budget, corpus growth, how the results
are consumed downstream -- and an automated verdict would hide that judgement
rather than inform it.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..evaluation.results import BenchmarkRun, StrategyResult

__all__ = [
    "build_run",
    "collect_environment",
    "quality_table",
    "latency_table",
    "summary_lines",
    "format_duration",
    "artifact_stem",
    "resolve_stem",
    "write_artifacts",
    "write_json",
    "render_markdown",
    "write_markdown",
    "print_report",
]

RERANKED_MARKER = "+ reranker"


def collect_environment() -> dict[str, Any]:
    """Facts about the machine that produced a result. Latency is meaningless
    without them."""
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:  # torch is only present once dense retrieval has been used
        import torch

        environment["torch"] = torch.__version__
        environment["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass
    return environment


def build_run(
    results: Sequence[StrategyResult],
    config: Config,
    *,
    corpus: dict[str, Any],
    dataset: dict[str, Any],
) -> BenchmarkRun:
    return BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        corpus=corpus,
        dataset=dataset,
        config=config.to_dict(),
        strategies=list(results),
        environment=collect_environment(),
    )


def _metric_header(key: str) -> str:
    """Abbreviated column headers.

    Spelling out ``Recall@10`` and ``nDCG@10`` for three cutoffs each pushes
    the table past 80 columns and rich then truncates a header to ``nDCG@...``,
    which is worse than an abbreviation the legend explains.
    """
    if key == "mrr":
        return "MRR"
    name, _, cutoff = key.partition("@")
    return f"{'R' if name == 'recall' else 'N'}@{cutoff}"


def format_duration(milliseconds: float) -> str:
    """Render a duration at a precision that stays informative across the four
    orders of magnitude this tool spans -- a BM25 query on a small corpus takes
    microseconds, loading a cross-encoder takes seconds."""
    if milliseconds >= 10_000:
        return f"{milliseconds / 1000:.1f}s"
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.2f}s"
    if milliseconds >= 10:
        return f"{milliseconds:.0f}ms"
    if milliseconds >= 1:
        return f"{milliseconds:.1f}ms"
    return f"{milliseconds:.2f}ms"


def quality_table(results: Sequence[StrategyResult], metric_keys: Sequence[str]) -> Table:
    table = Table(
        title="Retrieval quality (best per metric in bold)",
        title_justify="left",
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Strategy", no_wrap=True)
    for key in metric_keys:
        table.add_column(_metric_header(key), justify="right")

    best = {key: max(result.metrics[key] for result in results) for key in metric_keys}
    for result in results:
        row = [result.name]
        for key in metric_keys:
            value = result.metrics[key]
            text = f"{value:.3f}"
            row.append(f"[bold]{text}[/bold]" if value == best[key] else text)
        table.add_row(*row)
    return table


def latency_table(results: Sequence[StrategyResult]) -> Table:
    table = Table(
        title="Latency (Index is a one-off cost; the rest is per query)",
        title_justify="left",
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Index", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("Max", justify="right")
    for result in results:
        table.add_row(
            result.name,
            format_duration(result.index_time_ms),
            format_duration(result.latency.mean_ms),
            format_duration(result.latency.p50_ms),
            format_duration(result.latency.p95_ms),
            format_duration(result.latency.max_ms),
        )
    return table


def summary_lines(results: Sequence[StrategyResult], headline_metric: str) -> list[str]:
    """Named leaders on stated metrics -- not a recommendation."""
    if not results:
        return []

    # Prose spells the metric out; the abbreviation is a table convenience.
    label = "MRR" if headline_metric == "mrr" else headline_metric.replace("ndcg@", "nDCG@")
    lines: list[str] = []

    best_quality = max(results, key=lambda r: r.metrics[headline_metric])
    lines.append(
        f"Highest {label}: {best_quality.name} "
        f"({best_quality.metrics[headline_metric]:.3f}), "
        f"p95 {format_duration(best_quality.latency.p95_ms)}"
    )

    fastest = min(results, key=lambda r: r.latency.p95_ms)
    lines.append(
        f"Lowest p95 latency: {fastest.name} ({format_duration(fastest.latency.p95_ms)}), "
        f"{label} {fastest.metrics[headline_metric]:.3f}"
    )

    plain = [r for r in results if RERANKED_MARKER not in r.name]
    if plain and len(plain) != len(results):
        best_plain = max(plain, key=lambda r: r.metrics[headline_metric])
        if best_plain.name != best_quality.name:
            lines.append(
                f"Highest {label} without reranking: {best_plain.name} "
                f"({best_plain.metrics[headline_metric]:.3f}), "
                f"p95 {format_duration(best_plain.latency.p95_ms)}"
            )
    return lines


def artifact_stem(run: BenchmarkRun) -> str:
    """The filename stem for a run's artifacts, from its timestamp.

    Millisecond precision: at second precision two consecutive runs over a
    small corpus produced the same name and the later one silently overwrote
    the earlier one's results.
    """
    moment = datetime.fromisoformat(run.timestamp).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H%M%S") + f"{moment.microsecond // 1000:03d}Z"


def resolve_stem(run: BenchmarkRun, output_dir: str | Path) -> str:
    """A stem whose ``.json`` and ``.md`` names are both still free.

    Timestamps make collisions unlikely, not impossible, and losing a result
    silently is the one failure this tool must not have. Resolving once for
    both extensions is also what keeps a run's two artifacts named alike.
    """
    directory = Path(output_dir)
    base = artifact_stem(run)
    stem, attempt = base, 2
    while (directory / f"{stem}.json").exists() or (directory / f"{stem}.md").exists():
        stem = f"{base}-{attempt}"
        attempt += 1
    return stem


def write_artifacts(
    run: BenchmarkRun,
    output_dir: str | Path,
    metric_keys: Sequence[str],
    headline_metric: str,
) -> tuple[Path, Path]:
    """Write both artifacts of a run under one resolved name."""
    stem = resolve_stem(run, output_dir)
    return (
        write_json(run, output_dir, stem=stem),
        write_markdown(run, output_dir, metric_keys, headline_metric, stem=stem),
    )


def write_json(run: BenchmarkRun, output_dir: str | Path, *, stem: str | None = None) -> Path:
    """Write the machine-readable artifact, named by run timestamp."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem or artifact_stem(run)}.json"
    path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(align) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(
    run: BenchmarkRun,
    metric_keys: Sequence[str],
    headline_metric: str,
) -> str:
    """The same summary printed to the console, as a standalone Markdown
    document -- readable in an editor or GitHub, and diffable across runs,
    which the JSON artifact is not designed for."""
    results = run.strategies
    lines = [f"# RetrievEval report -- {run.timestamp}", ""]
    lines.append(f"- Corpus: `{run.corpus.get('path', '?')}` "
                 f"({run.corpus.get('documents', '?')} documents, "
                 f"{run.corpus.get('chunks', '?')} chunks)")
    lines.append(f"- Dataset: `{run.dataset.get('path', '?')}` "
                 f"({run.dataset.get('queries', '?')} labeled queries)")
    environment = run.environment
    env_bits = [f"Python {environment.get('python', '?')}", environment.get("platform", "")]
    if "cuda_available" in environment:
        env_bits.append("CUDA" if environment["cuda_available"] else "CPU only")
    lines.append(f"- Environment: {', '.join(bit for bit in env_bits if bit)}")
    lines.append("")

    best = {key: max(r.metrics[key] for r in results) for key in metric_keys}
    headers = ["Strategy", *[_metric_header(key) for key in metric_keys]]
    quality_rows = []
    for result in results:
        row = [result.name]
        for key in metric_keys:
            value = result.metrics[key]
            text = f"{value:.3f}"
            row.append(f"**{text}**" if value == best[key] else text)
        quality_rows.append(row)

    lines.append("## Retrieval quality (best per metric in bold)")
    lines.append("")
    lines.append(_markdown_table(headers, quality_rows))
    lines.append("")
    lines.append("R@k = Recall@k, N@k = nDCG@k, MRR = mean reciprocal rank")
    lines.append("")

    lines.append("## Latency (Index is a one-off cost; the rest is per query)")
    lines.append("")
    lines.append(
        _markdown_table(
            ["Strategy", "Index", "Avg", "p50", "p95", "Max"],
            [
                [
                    result.name,
                    format_duration(result.index_time_ms),
                    format_duration(result.latency.mean_ms),
                    format_duration(result.latency.p50_ms),
                    format_duration(result.latency.p95_ms),
                    format_duration(result.latency.max_ms),
                ]
                for result in results
            ],
        )
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    for line in summary_lines(results, headline_metric):
        lines.append(f"- {line}")
    lines.append("")
    lines.append(
        "*These numbers describe this corpus and this evaluation set only. "
        "Which trade-off to ship is your call.*"
    )
    return "\n".join(lines) + "\n"


def write_markdown(
    run: BenchmarkRun,
    output_dir: str | Path,
    metric_keys: Sequence[str],
    headline_metric: str,
    *,
    stem: str | None = None,
) -> Path:
    """Write the human-readable counterpart to :func:`write_json`, named by
    the same run timestamp so the two artifacts pair up in a listing."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem or artifact_stem(run)}.md"
    path.write_text(render_markdown(run, metric_keys, headline_metric), encoding="utf-8")
    return path


def print_report(
    console: Console,
    results: Sequence[StrategyResult],
    metric_keys: Sequence[str],
    headline_metric: str,
) -> None:
    console.print()
    console.print(quality_table(results, metric_keys))
    console.print("[dim]  R@k = Recall@k   N@k = nDCG@k   MRR = mean reciprocal rank[/dim]")
    console.print()
    console.print(latency_table(results))
    console.print()
    for line in summary_lines(results, headline_metric):
        console.print(f"  {line}")
    console.print()
    console.print(
        "[dim]These numbers describe this corpus and this evaluation set only. "
        "Which trade-off to ship is your call.[/dim]"
    )
