# Migrating Legacy Services to Nimbus

This guide covers moving a service from the legacy v1 virtual-machine platform to Nimbus. Plan for the migration to take two to four weeks per service.

## Step 1: Containerise

Write a Dockerfile based on the approved base image `nimbus/python:3.12-slim` (or `nimbus/go:1.23` for Go services). The container must run as a non-root user and listen on the port declared in the manifest.

## Step 2: Write the service manifest

Create `nimbus.yaml` in the repository root. It declares the service name, owning team, tier, port, CPU and memory requests, and the region. Run `nimbus validate` to check it before the first deploy.

## Step 3: Move secrets to Vault

Copy every credential from the legacy configuration files into Vault and delete them from the old hosts. Reference them from the manifest; never bake them into the image.

## Step 4: Database migration

Provision a Nimbus PostgreSQL cluster in the same region as the service, then move the data. Small databases can be copied with dump and restore; databases that cannot take downtime should use logical replication and cut over when lag is below one second.

## Step 5: Observability

Expose `/metrics`, ship JSON logs to stdout and define at least an availability alert and a latency alert before taking production traffic.

## Step 6: Canary launch

Route traffic to the Nimbus deployment through the standard canary process. Keep the v1 service running in parallel so traffic can be moved back.

## Step 7: Decommission v1

After 14 days of stable production traffic on Nimbus, stop the v1 virtual machines and archive their configuration. Mark the service as migrated in the service catalogue.

## Regions and data residency

Nimbus runs in two regions, `eu-west-1` and `us-east-1`. Services that store personal data of EU customers must run their database in `eu-west-1`.
