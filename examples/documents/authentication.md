# Authentication

The API authenticates requests with a bearer token sent in the
`Authorization` header. Tokens are issued per workspace and inherit the
permissions of the member who created them.

## Token lifetime

An access token is valid for 60 minutes. After that the API rejects requests
with HTTP 401 and the error code `TOKEN_EXPIRED`. Use the refresh token to
obtain a new access token without asking the user to sign in again.

## Refresh tokens

A refresh token is valid for 30 days and is rotated on every use: the response
to a refresh request contains a new refresh token, and the previous one stops
working immediately. Reusing an old refresh token returns
`REFRESH_TOKEN_REUSED` and revokes the whole token family as a precaution.

## Signing out

Signing out revokes the refresh token but leaves any already-issued access
token valid until it expires. For an immediate cut-off, revoke the token
family explicitly through the revocation endpoint.
