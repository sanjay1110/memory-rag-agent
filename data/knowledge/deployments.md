# Nimbus Deployments

Nimbus is the internal application platform. Every service on Nimbus ships through the same release pipeline, and this document describes how that pipeline behaves.

## Release pipeline

A release moves through three stages: build, canary and full rollout. The build stage produces an immutable container image tagged with the git commit SHA. Images are scanned for known vulnerabilities and a critical finding blocks the release.

Start a release with `nimbus deploy --env prod --image <sha>`. Staging releases use `--env staging` and skip the canary stage.

## Canary analysis

During the canary stage, 5% of production traffic is routed to the new release for 15 minutes. The canary analyser compares error rate and p99 latency against the current stable release. If the canary error rate exceeds 2% or p99 latency regresses by more than 20%, the release is automatically rolled back and the deploying engineer is notified in Slack.

A healthy canary is promoted to 100% of traffic in four steps of 25% each, two minutes apart.

## Rollback

To roll back manually, run `nimbus rollback <service> --to <release-id>`. Rollback re-points traffic to a previous image and completes in under one minute because images are never rebuilt. The last ten releases of each service are kept available for rollback.

Database schema changes are not reverted by a rollback, which is why schema migrations must be backward compatible (see the database guide).

## Deployment windows

Production deploys are blocked on Fridays after 14:00 UTC and during the year-end freeze from December 20 to January 3. Emergency fixes during a blocked window need approval from the on-call incident commander, recorded with `nimbus deploy --override-reason`.

## Stateful services

Stateful services such as queues and caches use blue/green deployment instead of canaries. The new (green) environment is warmed up fully before traffic switches, and the old (blue) environment is kept for 24 hours to allow a fast switch back.
