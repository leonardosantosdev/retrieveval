# Rate limits

Every workspace has a quota of 600 requests per minute across all tokens.
Bulk endpoints count each item in the payload as one request against the
quota, so a batch of 50 items consumes 50 units.

## Reading your current usage

Each response carries `X-RateLimit-Remaining` and `X-RateLimit-Reset`. Read
these headers rather than counting requests yourself; retries, redirects and
internal fan-out all consume quota.

## Raising the limit

Higher quotas are available on the Scale plan. Requests for a temporary
increase ahead of a migration are usually granted within one business day.

## Designing around the limit

Spread scheduled jobs across the hour rather than starting them all on the
minute. A client-side token bucket sized slightly below the quota is more
effective than reacting to 429 responses after the fact.
