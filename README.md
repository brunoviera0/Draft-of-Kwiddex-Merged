# Kwiddex

A tool to aid with forensic document examination, with analysis powered by deep learning, digital certification, verification, and document comparison.

Live at: https://kwiddex.com

## Navigation

Home, Certify, Verify, Compare, About

Desktop shows full nav bar. Mobile shows hamburger menu with dropdown that auto-closes on route change.

## Features

### Certify (Upload is public, certification requires login)

Upload a PDF for deep learning analysis using a ResNet18 CNN with Monte Carlo inference (30 augmented samples). Returns confidence score, 95% confidence interval, agreement rate, and standard deviation. No documents are stored.

After reviewing results, authenticated users can certify the document. Certification embeds an RSA-2048 signed certificate in the PDF metadata, appends a visible certificate page, and stores the document hash in Google Cloud Datastore. The certified PDF is returned for download.

### Verify (Public)

Upload a PDF to check if it contains a valid Kwiddex certificate. The system extracts the embedded certificate and RSA signature, validates the signature, compares the file hash against the stored hash to detect modifications, and checks Datastore for revocation status and disputes. Displays a plain language verdict: certified and unmodified, modified after certification, revoked, invalid signature, or no certificate found.

### Compare (Public)

Upload two document images for multi-region forensic comparison. The V4 engine normalizes both images to a standard size, aligns them using cross-correlation search, then analyzes artwork, text, borders, and print texture independently. Produces a weighted final score with verdict (likely real, suspicious, or likely fake), difference heatmap, and zoomed micro-region comparisons. Runs entirely client-side.

### Certificate Dispute System (Requires login to report)

Any authenticated user can dispute a certificate with a written explanation (minimum 50 characters). Disputes are public, permanent, informational annotations that do not block or invalidate the certificate. One report per user per certificate. Users cannot dispute their own certificates.

The original certifier can dismiss a dispute with a written response (minimum 20 characters) or accept it, which triggers a self-revoke. Dismissed disputes remain permanently visible. Kwiddex does not adjudicate disputes. The dispute record serves as the reliability marker.

### Certificate Revocation (Requires login)

Only the original certifier can revoke their own certificate. Revocation is permanent. Revoked certificates fail verification.

## Architecture

Single GCP VM running three systemd services that auto-start on boot.

Frontend: React/Vite static build served by Nginx.

Express BFF (port 3001): Proxies API requests to FastAPI. Validates Auth0 tokens on protected routes.

FastAPI (port 8000): CNN inference, RSA certificate generation, verification, dispute system, Datastore operations. Runs on Uvicorn.

Nginx: SSL termination via Let's Encrypt, reverse proxy routing (/api/* to Express, /ml/* to FastAPI), rate limiting, security headers.

## Auth Flow

Frontend redirects to Auth0 Universal Login. Auth0 issues RS256 JWTs. Frontend attaches tokens as Bearer headers on protected requests. Both Express and FastAPI validate tokens using JWKS public keys.

Protected endpoints: /certify, /revoke-certificate, /report-certificate, /resolve-dispute, /my-certificates

Public endpoints: /predict, /monte_carlo, /verify-certificate, /health, /api/physical/score

## Security

Authentication: Auth0 Universal Login with RS256 JWTs validated via JWKS.

Rate Limiting: FastAPI endpoints rate limited via slowapi (5 to 10 requests per minute per IP depending on endpoint). Nginx applies global rate limiting at approximately 10 requests per second per IP with burst.

File Size Limits: 50MB across Express, FastAPI, and Nginx.

CORS: Restricted to https://kwiddex.com, https://www.kwiddex.com, and localhost for development.

Security Headers (Nginx): HSTS (1 year), X-Frame-Options (SAMEORIGIN), X-Content-Type-Options (nosniff), X-XSS-Protection, Referrer-Policy (strict-origin-when-cross-origin).

Disabled Endpoints: /register and /login return 410 (handled by Auth0).

Data Storage: No documents or PII stored. Datastore holds metadata only: prediction results, certification records, and disputes. RSA private key is in .gitignore and must be backed up securely. If lost, existing certificates become unverifiable.

## Tests

### Full Stack Tests (tests/test_full_stack.sh)

40+ assertions across 9 sections: health, security headers, auth/access control, public endpoints, input validation, rate limiting, CORS, certificate roundtrip, and SSL.

### FastAPI Unit Tests (backend/unit_tests_api.py)

29 tests covering health, disabled endpoints, protected endpoint auth, CNN prediction, Monte Carlo inference, verification, disputes, input validation, and unknown routes. Requires GCP Datastore credentials (runs locally on VM only).

### Cypress E2E Tests (frontend/cypress/e2e/)

4 spec files covering auth flows, navigation, page content, and upload functionality.

### Dispute Flow Demo (tests/test_dispute_flow.sh)

End-to-end demo of the certificate dispute lifecycle. Requires two different Auth0 accounts for the full flow.

### GitHub Actions CI

Runs on push to main: full stack tests and Cypress E2E in parallel. Unit tests excluded from CI (require GCP credentials).

## Running Locally

FastAPI:
cd backend
uvicorn predict:app --host 0.0.0.0 --port 8000

Express:
cd server
npm install && npm run dev:api

Frontend:
cd frontend
npm install && npm run dev
