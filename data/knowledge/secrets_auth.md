# Secrets and Authentication

## Secret storage

All secrets live in Vault under the path `secret/nimbus/<team>/<service>`. Secrets are injected into containers at start-up as files under `/run/secrets`, never as plain environment variables and never committed to `.env` files or the repository. The secret scanner blocks any commit that contains a credential pattern.

## Rotation

Secrets must be rotated at least every 90 days. Database credentials are rotated automatically by Vault every 30 days through dynamic credentials, so services must re-read the credential file on authentication failure rather than caching it forever.

## Service-to-service authentication

Services authenticate to each other with mutual TLS. Certificates are issued by the platform certificate authority, are valid for 24 hours and are renewed automatically by the sidecar. A service may only call another service when the callee's access policy lists it.

## User authentication

Human users sign in through single sign-on using OpenID Connect. Access tokens expire after 1 hour and refresh tokens after 24 hours. Admin actions in the Nimbus console require a second factor, re-checked every 12 hours.
