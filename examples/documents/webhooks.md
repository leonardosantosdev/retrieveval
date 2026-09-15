# Webhooks

Webhooks deliver events to an HTTPS endpoint you control. Configure them under
**Account → Developers → Webhooks**.

## Delivery and retries

A delivery is considered successful when your endpoint answers with a 2xx
status within 10 seconds. Failed deliveries are retried with exponential
backoff for up to 24 hours, after which the event is dropped and the endpoint
is marked unhealthy.

## Verifying the signature

Each request carries an `X-Signature` header holding an HMAC-SHA256 of the raw
body using your endpoint's signing secret. Compare it against the raw bytes of
the body, before any JSON parsing, or the signature will not match.

## Ordering

Events are not guaranteed to arrive in order. Each payload includes a
monotonic `sequence` field; ignore an event whose sequence is lower than one
you have already processed for the same resource.
