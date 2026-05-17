# OAuth 2.0 Authentication Flow

## Overview

Our authentication system implements the **OAuth 2.0 Authorization Code Flow with PKCE** for all client applications. This document describes the current authentication architecture.

## Flow

1. Client generates `code_verifier` and `code_challenge` (S256)
2. Client redirects to `/authorize` with `code_challenge`
3. User authenticates via login form
4. Authorization server issues `authorization_code`
5. Client exchanges code + `code_verifier` for tokens at `/token`
6. Access token (JWT, 15min TTL) + Refresh token (opaque, 7d TTL) returned

## Token Structure

```json
{
  "sub": "user_123",
  "iss": "auth.payments.internal",
  "aud": "payments-api",
  "exp": 1704067200,
  "scope": "payments:read payments:write",
  "tenant_id": "acme_corp"
}
```

## Endpoints

- `POST /authorize` — initiate authorization
- `POST /token` — exchange code for tokens
- `POST /token/refresh` — refresh access token
- `POST /token/revoke` — revoke refresh token
- `GET /userinfo` — get authenticated user info
- `GET /.well-known/openid-configuration` — OIDC discovery

## Security

- All tokens signed with RS256 (2048-bit RSA keys)
- Key rotation every 90 days
- Refresh token rotation enabled (one-time use)
- Rate limiting: 10 requests/minute per IP on auth endpoints
