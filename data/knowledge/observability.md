# Observability

## Metrics and dashboards

Every service exposes Prometheus metrics on port 9090 at `/metrics`. Nimbus generates a default Grafana dashboard per service showing request rate, error rate and latency percentiles.

## Logging

Logs are written to stdout as JSON and collected into Loki. Logs are retained for 30 days. Personal data must not be logged; the log pipeline masks e-mail addresses and card numbers but teams remain responsible for what they emit.

## Tracing

Distributed tracing uses OpenTelemetry. The default sampling rate is 10% of requests, and 100% of requests that end in an error are kept.

## Service level objectives

Tier-1 services must meet a p99 latency objective below 300 milliseconds and 99.9% monthly availability. Tier-2 services target p99 below 800 milliseconds and 99.5% availability.

## Alerting

Alerts are defined as code in the service repository. Severity 1 and severity 2 alerts page the on-call engineer through PagerDuty; severity 3 alerts create a ticket instead of paging.
