The purpose of this repository is to test the integration of Kwiddex. From the following repositories:

CNN Model/FastAPI backend: https://github.com/brunoviera0/Kwiddex-CNN-Model

Express Server/Frontend: https://github.com/Kwiddex/kwiddex


Server Architecture
-------------------
 
Express BFF (port 3001) sits between the frontend and FastAPI (port 8000).
Express proxies scoring requests to FastAPI, handles PDF metadata extraction,
and protects the email endpoint with Auth0 JWT validation.
 
 
 
Navigation: Home, Analyze, Sign, Verify, Compare, About.
Desktop shows full nav bar. Mobile shows hamburger menu with
dropdown. Menu auto-closes on route change.
 
 
 
Auth flow: Frontend redirects to Auth0 Universal Login. Auth0 issues RS256
JWTs with audience https://api.kwiddex.com. Frontend stores tokens via
Auth0 SDK and attaches them as Bearer tokens on protected requests.
FastAPI validates tokens using auth0_validator.py which fetches JWKS
public keys from Auth0. Express validates tokens on the email endpoint
using express-oauth2-jwt-bearer.
 
 
 
Scoring flow: Frontend uploads a file to Express /api/physical/score.
Express forwards it to FastAPI /monte_carlo. FastAPI runs 30 augmented
CNN inferences and returns classification + stats. Express maps the
response into the shape the frontend expects (confidence, confidence
interval, Monte Carlo stats).
 
 
 
Certification flow: Frontend sends file + Auth0 Bearer token to
FastAPI /certify via Nginx /ml/certify. FastAPI validates the JWT,
runs Monte Carlo CNN analysis, creates an RSA signed certificate,
embeds it in the PDF metadata, stores the certification record and
certified file hash in Google Cloud Datastore, and returns the
certified PDF as a downloadable file.
 
 
 
Verification flow: Frontend sends PDF to FastAPI /verify-certificate
via Nginx /ml/verify-certificate. FastAPI extracts the embedded
certificate and RSA signature from PDF metadata, verifies the
signature using the public key, checks Datastore for revocation
status, and compares the file hash against the stored certified
hash to detect post certification modifications.
 
 
 
Running the Servers
-------------------
 
FastAPI:
  
  cd backend
  
  uvicorn predict:app --host 0.0.0.0 --port 8000
 
Express:
  
  cd server
  
  echo "FASTAPI_URL=http://localhost:8000" > .env
  
  npm install
  
  npm run dev:api
 
Frontend:
  
  cd frontend
  
  npm install
  
  npm run dev
 
 
Auth0 Configuration
-------------------
 
  Domain: dev-jamm61acuiu8yfq6.us.auth0.com
  
  Client ID: yBA5UqZj65KlgdyHsmQs4RycS4JqN8jf
  
  Audience: https://api.kwiddex.com
  
  Allowed Callback URLs: https://kwiddex.com
  
  Allowed Logout URLs: https://kwiddex.com
  
  Allowed Web Origins: https://kwiddex.com
 
  The Kwiddex API must be registered in Auth0 Dashboard under
  Applications > APIs with identifier https://api.kwiddex.com
  and the application must be authorized under both User Access
  and Client Access tabs.
 
  Google social connection uses custom OAuth credentials.
  To transfer ownership, create new Google OAuth credentials
  in Google Cloud Console with redirect URI
  https://dev-jamm61acuiu8yfq6.us.auth0.com/login/callback
  and replace the Client ID and Secret in Auth0 Dashboard
  under Authentication > Social > Google.
 
 
Features
--------
 
Analyze (Public, no login required)
 
  Upload a PDF or image to run it through the ResNet18 CNN with
  Monte Carlo inference (30 augmented samples). Returns confidence
  score, 95% confidence interval, agreement rate, and standard
  deviation. No documents are stored. Metadata only is recorded
  in Datastore.
 
 
Certify (Requires Auth0 login)
 
  Two step flow. Step 1: upload a PDF for Monte Carlo CNN analysis.
  User reviews the results and a disclaimer that certification is
  their professional judgment. Step 2: user clicks Certify This
  Document. The PDF is certified with a Kwiddex RSA signed
  certificate embedded in the PDF metadata. A visible certificate
  page is appended. The certified PDF hash is stored in Datastore
  for integrity checking. The certified PDF is returned for download.
  The certificate records: unique ID, document hash, Monte Carlo
  confidence score, reviewer email, timestamp, and status.
 
 
Verify (Public, no login required)
 
  Upload a PDF to check if it contains a valid Kwiddex certificate.
  Extracts the certificate and RSA signature from PDF metadata,
  verifies the signature, checks Datastore for revocation, and
  compares the file hash against the stored certified hash to
  detect modifications. Also extracts PDF metadata (title, author,
  creator, producer, dates) and detects known editors. Displays
  plain language verdict: certified and unmodified, modified after
  certification, revoked, invalid signature, or no certificate found.
 
 
Compare (Public, no login required)
 
  Two tabs: Compare and Extract Text.
  
  Compare tab: upload two document images for side by side spectral
  analysis using the LWSP (Linear Wave Stochastic Process) engine.
  Images are converted to grayscale, band pass filtered, transformed
  via 2D FFT, and compared using Power Spectral Density cross
  correlation. Returns a similarity score from 0 to 100. User
  controllable high cut filter slider adjusts how much fine detail
  is included. Supports zoom, pan, notes, case naming, multi entry,
  and PNG report download. Runs entirely client side with no backend.
  
  Extract Text tab: upload a PDF to extract embedded text content
  using pdfjs-dist. Works with native text PDFs. Copy extracted text
  to clipboard.
 
 
Certificate Revocation (Requires Auth0 login)
 
  POST to /ml/revoke-certificate/{certificate_id} sets the certificate
  status to revoked in Datastore. Only the original certifier can
  revoke their own certificate. Revoked certificates fail verification.
  Revocation is permanent. Does not invalidate the file, only the
  certificate.


Certificate Dispute System (Requires Auth0 login to report)
 
  Any authenticated user can dispute a certificate by submitting a
  report with a written reason (minimum 50 characters). Disputes are
  public, permanent, informational annotations. They do not block or
  invalidate the certificate. Anyone who verifies a disputed document
  sees the dispute details including the reporter, reason, and status.

  One report per user per certificate. Users cannot dispute their own
  certificates. The original certifier can dismiss a dispute (with a
  required written response, minimum 20 characters) or accept it
  (which triggers a self-revoke). Dismissed disputes remain permanently
  visible with the certifier's response.

  Kwiddex does not adjudicate disputes. Resolution occurs through
  established professional or legal channels. The dispute record
  serves as the reliability marker.

  Endpoints:
    POST /report-certificate/{certificate_id} - file a dispute
    POST /resolve-dispute/{certificate_id}/{dispute_id} - dismiss or accept
    GET /my-certificates - view certificates signed by the current user

  Datastore Kind: CertificateDispute
 
 
Current Tests
-------------
 
Full Stack Tests (tests/test_full_stack.sh)
 
  40+ assertions across 9 sections: health, security headers,
  auth/access control, public endpoints, input validation,
  rate limiting, CORS, certificate roundtrip, and SSL.
  Includes authenticated certificate roundtrip when AUTH_TOKEN
  is set.
 
  Run:
 
    export AUTH_TOKEN="your_auth0_token"
 
    bash tests/test_full_stack.sh
 
 
FastAPI Unit Tests (backend/unit_tests_api.py)
 
  29 tests covering health, disabled endpoints (register/login
  return 410), protected endpoint auth checks (certify, revoke,
  document, certificate, my-certificates, report-certificate,
  resolve-dispute), CNN prediction, Monte Carlo inference,
  verify-certificate (including dispute fields), input
  validation, and unknown routes. Requires GCP Datastore
  credentials so runs locally on the VM only, not in CI.
 
  Run:
 
    cd backend
 
    python -m unittest unit_tests_api.py -v
 
 
Cypress Tests (frontend/cypress/e2e/)
 
  4 spec files, approximately 17 tests:
    auth.cy.js - Sign in button, public page access
    navigation.cy.js - Home page, nav links, feature cards, redirects
    scoring.cy.js - CNN upload and Monte Carlo results
    verify.cy.js - Upload prompt, dispute description, certificate check
 
  Run locally:
 
    cd frontend && npx cypress run
 
 
Dispute Flow Demo (tests/test_dispute_flow.sh)
 
  End-to-end demo of the certificate dispute lifecycle:
  certify, verify clean, file dispute, verify with dispute
  visible, check certifier profile, dismiss dispute, verify
  permanent audit trail. Requires AUTH_TOKEN. Note: M2M token
  is the same identity as certifier so self-report is blocked.
  Full flow requires two different Auth0 accounts.
 
  Run:
 
    export AUTH_TOKEN="your_auth0_token"
 
    bash tests/test_dispute_flow.sh
 
 
Security
--------
 
Authentication
 
  Auth0 Universal Login with RS256 JWTs. AuthGuard component
  on the frontend gates the Sign/Certify page. FastAPI validates
  tokens via JWKS public keys. Express validates tokens on the
  email endpoint via express-oauth2-jwt-bearer.
 
  Protected endpoints: /certify, /revoke-certificate,
  /certificate/{id}, /document/{id}, /api/email,
  /report-certificate/{id}, /resolve-dispute/{id}/{id},
  /my-certificates.
 
  Public endpoints: /predict, /monte_carlo, /verify-certificate,
  /health, /api/verify, /api/physical/score.
 
 
Rate Limiting
 
  FastAPI endpoints are rate limited via slowapi:
 
    /predict: 10 requests per minute per IP
    /monte_carlo: 5 requests per minute per IP
    /certify: 5 requests per minute per IP
    /verify-certificate: 10 requests per minute per IP
 
  Express email endpoint is rate limited by IP via custom limiter.
 
  Nginx applies global rate limiting at approximately
  10 requests per second per IP with burst allowance.
 
 
File Size Limits
 
  Express: 50MB via multer configuration.
  FastAPI: 50MB enforced in application code.
 
 
CORS
 
  FastAPI: restricted to https://kwiddex.com, https://www.kwiddex.com,
  and localhost origins for development.
 
  Express: restricted to production and local development origins.
 
 
Security Headers (Nginx)
 
  Strict Transport Security (HSTS, 1 year with subdomains)
  X Frame Options (SAMEORIGIN)
  X Content Type Options (nosniff)
  X XSS Protection (enabled, block mode)
  Referrer Policy (strict origin when cross origin)
 
 
Disabled Endpoints
 
  /register returns 410 (handled by Auth0)
  /login returns 410 (handled by Auth0)
 
 
Data Storage
 
  No user documents or PII are stored. Google Cloud Datastore
  holds metadata only: prediction results (confidence, label, UUID),
  certification records (certificate ID, document hash, certified
  file hash, reviewer, status). RSA key pair at backend/keys/
  (private key signs, public key verifies). Private key is in
  .gitignore and must be backed up securely. If lost, all existing
  certificates become unverifiable.
 
 
Infrastructure
--------------
 
Live at: https://kwiddex.com
 
  Single GCP VM (sentiment-prod)

 
Services
 
  Nginx (port 80/443): reverse proxy, SSL termination, static frontend
  Express (port 3001): BFF, email, PDF metadata
  FastAPI (port 8000): CNN model, certificates, auth validation
  All services auto start on boot via systemd.
 
Nginx Routing
 
  / serves static frontend build
  /api/* proxies to Express on port 3001
  /ml/* proxies to FastAPI on port 8000 (prefix stripped)
 
SSL
 
  Certificate: Let's Encrypt via certbot
  Auto renews every 60 days (cron job)
  Manual renewal if needed: sudo certbot renew
 
Google Cloud Datastore
 
  Kinds: PredictionResult, Certification, CertificateDispute
  No file storage. Metadata only.
 
 
### Useful Commands
```
#Check service status
sudo systemctl status kwiddex-fastapi kwiddex-express kwiddex-frontend
 
#View logs
sudo journalctl -u kwiddex-fastapi -f
sudo journalctl -u kwiddex-express -f
 
#Restart after code changes
sudo systemctl restart kwiddex-fastapi kwiddex-express
 
#Rebuild frontend after changes
cd frontend && npm run build
 
#Renew SSL manually
sudo certbot renew
 
#Test Nginx config
sudo nginx -t && sudo systemctl reload nginx
```
 
GitHub Actions CI (.github/workflows/ci.yml)
 
  Triggers on push to main, pull requests to main, and manual
  dispatch. Two jobs run in parallel:
    Backend: full-stack tests + authenticated certificate roundtrip
    Cypress: 4 spec files headless in Chrome against live site
  Unit tests are excluded from CI (require GCP Datastore credentials).
 
 
### Config Files
```
Nginx config and systemd service files are stored in docs/
```
 
 
