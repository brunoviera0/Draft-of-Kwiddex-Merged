"""
Auth0 RS256 JWT validation for FastAPI.
Replaces the legacy HS256 shared-secret validation in auth.py.
"""

import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache

AUTH0_DOMAIN = "dev-jamm61acuiu8yfq6.us.auth0.com"
AUTH0_AUDIENCE = "https://api.kwiddex.com"
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/"
JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# Cache the JWKS client (it handles key rotation internally)
@lru_cache()
def get_jwks_client():
    return PyJWKClient(JWKS_URL, cache_keys=True)

security = HTTPBearer(auto_error=False)


def validate_auth0_token(token: str) -> dict:
    """Validate an Auth0-issued RS256 JWT and return the payload."""
    try:
        jwks_client = get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired.")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience.")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid token issuer.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI dependency — enforces Auth0 JWT on protected endpoints."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide a Bearer token."
        )
    return validate_auth0_token(credentials.credentials)
