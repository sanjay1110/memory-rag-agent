# Incident Management

## Severity levels

Sev1 is a customer-facing outage or data loss and must be acknowledged within 5 minutes. Sev2 is a significant degradation with a workaround and must be acknowledged within 15 minutes. Sev3 is a minor issue handled on the next business day.

## Roles

Every Sev1 and Sev2 incident has an incident commander who coordinates the response, a communications lead who updates stakeholders, and subject-matter responders. The incident commander, not the most senior engineer, makes the call on mitigation steps.

## Communication

For Sev1 incidents the public status page is updated at least every 30 minutes until resolution. Internal updates go to the `#incidents` channel.

## Postmortems

A blameless postmortem is required for every Sev1 and Sev2 incident and must be published within 5 business days. It records the timeline, root cause, contributing factors and action items with owners and due dates.
