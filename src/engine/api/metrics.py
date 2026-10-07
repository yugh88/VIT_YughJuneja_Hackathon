"""Prometheus metrics, on a per-app registry so tests can build many apps."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.docs_ingested = Counter(
            "sentinel_docs_ingested_total", "Raw documents received", ["source"], registry=self.registry
        )
        self.signals_emitted = Counter(
            "sentinel_signals_total", "Signals published", ["event"], registry=self.registry
        )
        self.dead_letters = Counter(
            "sentinel_dead_letter_total",
            "Documents that failed processing",
            ["stage"],
            registry=self.registry,
        )
        self.duplicates = Counter(
            "sentinel_duplicates_total", "Near-duplicate documents collapsed", registry=self.registry
        )
        self.bus_lag = Gauge(
            "sentinel_bus_lag", "Messages waiting per topic", ["topic"], registry=self.registry
        )
        self.source_up = Gauge(
            "sentinel_source_up",
            "1 if the source delivered recently",
            ["source", "mode"],
            registry=self.registry,
        )
        self.pipeline_latency = Histogram(
            "sentinel_pipeline_latency_seconds",
            "Ingest-to-signal processing latency",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
