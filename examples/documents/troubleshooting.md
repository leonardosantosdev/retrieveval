# Troubleshooting API errors

## HTTP 401 responses

A 401 with `TOKEN_EXPIRED` means the access token passed its 60 minute
lifetime; request a new one with your refresh token. A 401 with
`TOKEN_INVALID` means the token was malformed, revoked, or issued for a
different workspace. Check that you are not sending a token from a staging
workspace to the production API.

## HTTP 429 responses

Error code `RATE_LIMIT_EXCEEDED` means you exceeded the request quota for the
current minute. The `Retry-After` header tells you how many seconds to wait.
Retrying immediately makes the situation worse because rejected requests still
count against the quota.

## HTTP 500 responses

`INTERNAL_ERROR` responses are safe to retry with exponential backoff. If the
same request fails three times in a row, capture the `X-Request-Id` header
from the response and include it in a support ticket.

## Connection resets

Repeated connection resets are usually a proxy timing out a long-running
request. Requests that take more than 30 seconds should use the asynchronous
job endpoints instead.
