# Nimbus Databases

Nimbus provides managed PostgreSQL 15 clusters. Teams should not run their own database servers on Nimbus.

## Provisioning

Create a database with `nimbus db create <name> --size medium --region <region>`. Sizes are small (2 vCPU), medium (4 vCPU) and large (16 vCPU). Every cluster has one primary and one synchronous replica in a different availability zone.

## Connection pooling

All connections go through PgBouncer in transaction pooling mode. The pool size per service instance is controlled by the environment variable `NIMBUS_DB_POOL`, which defaults to 20 connections and may be raised to a maximum of 100. Long-lived session features such as advisory locks and prepared statements across transactions are not supported through the pooler.

## Backups and restore

Automated backups run nightly at 02:00 UTC and are retained for 14 days. Point-in-time recovery is available for the last 7 days. To restore, run `nimbus db restore <name> --snapshot <snapshot-id>` or `--to-time <timestamp>` for point-in-time recovery. A restore always creates a new cluster; the original is left untouched so data can be compared before cutting over.

## Schema migrations

Schema migrations run with `nimbus db migrate` as a separate step before the application deploy. Migrations must follow the expand/contract pattern: first add new columns or tables in a backward-compatible way, deploy the code that uses them, and only remove old columns in a later release. Each migration statement runs with a lock timeout of 5 seconds so a blocked migration fails fast instead of stalling production traffic.

## Moving data into Nimbus

For data imports below 50 GB, use `pg_dump` and `pg_restore` during a maintenance window. Larger databases, or services that cannot accept downtime, should use logical replication from the old database into the Nimbus cluster and cut over once replication lag is under one second.
