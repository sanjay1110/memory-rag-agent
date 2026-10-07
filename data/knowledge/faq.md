# Platform FAQ

Frequently asked questions collected from the `#platform-help` channel. Some answers repeat material from the main guides on purpose; the agent's de-duplication step is expected to collapse them.

## How do I roll back a bad release?

To roll back manually, run `nimbus rollback <service> --to <release-id>`. Rollback re-points traffic to a previous image and completes in under one minute because images are never rebuilt.

## Why was my deploy blocked on a Friday afternoon?

Production deploys are blocked on Fridays after 14:00 UTC and during the year-end freeze from December 20 to January 3.

## Can I use spot instances for my API?

No. Spot instances are allowed for batch and CI workloads only, because they can be reclaimed at short notice.

## Who do I ask about the on-call rota?

The rota is owned by each team; the platform team only runs the paging integration.
