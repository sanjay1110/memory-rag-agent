# API Gateway Rate Limits

## Default limits

The Nimbus API gateway applies a default limit of 1000 requests per minute per client key, with a burst allowance of 200 requests. Limits are enforced per region, so a client using two regions has an independent budget in each.

## Throttling behaviour

When a client exceeds its limit the gateway returns HTTP 429 with error code `E429` and a `Retry-After` header in seconds. Clients should back off exponentially with jitter, starting from the `Retry-After` value. Repeated violations for more than 10 minutes trigger an alert to the owning team.

## Requesting a higher quota

To raise a limit, open a ticket in the `#platform-quota` Slack channel with the expected peak request rate and a load-test result. Quota increases are reviewed within 2 business days. Increases above 10,000 requests per minute require a capacity review with the platform team.

## Per-tenant quotas

Multi-tenant services can define per-tenant quotas in `nimbus.yaml` under `gateway.tenant_limits` so that a single noisy tenant cannot exhaust the shared limit.
