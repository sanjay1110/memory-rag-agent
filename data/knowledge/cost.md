# Cost Management

## Team budgets

Each team has a monthly cloud budget. An alert is sent to the team channel when spend reaches 80% of the budget, and a spend review with engineering finance is scheduled when it reaches 100%.

## Required tags

Every resource must carry the tags `team`, `service` and `env`. Untagged resources are reported daily and are deleted after 14 days if still untagged.

## Saving money

Spot instances may be used for batch and CI workloads only, never for latency-sensitive services. Preview and development environments are shut down automatically after 72 hours without traffic. Storage classes move objects to infrequent-access storage after 30 days without reads.
