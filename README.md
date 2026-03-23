The purpose of this repository is to test the integration of Kwiddex. From the following repositories:

CNN Model/FastAPI backend: https://github.com/brunoviera0/Kwiddex-CNN-Model

Express Server/Frontend: https://github.com/Kwiddex/kwiddex


2/26/26

Server Architecture
-------------------

Express BFF (port 3001) sits between the frontend and FastAPI (port 8000).
Express proxies auth and scoring requests to FastAPI, verifies JWTs locally,
and handles PDF metadata extraction on its own.



Auth flow: Frontend sends {email, password} to Express. Express maps email
to username and forwards to FastAPI /register or /login. FastAPI issues a
JWT (HS256). Express verifies that JWT using the shared KWX_JWT_SECRET.



Scoring flow: Frontend uploads a file to Express /api/physical/score.
Express forwards it to FastAPI /monte_carlo. FastAPI runs 30 augmented
CNN inferences and returns classification + stats. Express maps the
response into the shape the frontend expects (score, reasons,
flags, suggestions, subscores, confidence).

***Response shape was changed to fit existing frontend, will be changed back to (score (%), Confidence Interval (CI))***



Running the Servers
-------------------

FastAPI:
  
  cd backend
  
  export KWX_JWT_SECRET=
  
  uvicorn predict:app --host 0.0.0.0 --port 8000

Express:
  
  cd server
  
  echo "KWX_JWT_SECRET=" > .env
  
  echo "FASTAPI_URL=http://localhost:8000" >> .env
  
  npm install
  
  npm run dev:api


Current Tests
-------------

FastAPI Unit Tests (backend/unit_tests_api.py)
   
   56 tests across 13 test classes. Tests FastAPI endpoints in isolation:
   auth (register, login, token creation/verification), CNN prediction,
   Monte Carlo inference, PDF certification and verification, certificate
   storage/lookup/revocation, and input validation. Does not require
   Express to be running.

   Run:
     
     cd backend
     
     python -m unittest unit_tests_api.py -v

Express-FastAPI Integration Tests (tests/test_express_fastapi.sh)
   
   38 assertions across 7 groups. Tests the live connection between
   Express and FastAPI with both services running. Covers:

   Health: Both services reachable, model loaded, RSA keys
   present, Express reports CNN provider.

   Auth Proxy: Signup via Express creates user in FastAPI,
   login returns correct response shape {token, user: {id, email}},
   duplicate signup rejected, wrong password rejected, empty fields
   rejected.

   JWT Interop: Token from Express login works on /auth/me.
   Token from direct FastAPI login also works on Express /auth/me
   (proves shared secret matches). Missing, invalid, and expired
   tokens all return 401.

   CNN Scoring: FastAPI /predict and /monte_carlo work
   directly. Express /physical/score full pipeline returns all
   required fields (score 0-100, reasons, flags, suggestions,
   confidence, analysisId, subscores). Provider is "cnn".

   ***as mentioned above, required fields will be changed to just encompass score and confidence intervals**

   PDF Metadata: Express /verify extracts sha256 and core
   metadata from uploaded PDFs.

   WordPress Proxy: Returns 503 gracefully when
   WORDPRESS_URL is not configured. (awaiting wordpress integration)

   Error Handling: Missing file returns 400, unknown routes
   return 404, non-image files rejected, empty email rejected.

   Run:
     
     bash tests/test_express_fastapi.sh


Next Week
---------

-Verify front end integration with Cypress end-to-end tests covering auth flows, document
upload and scoring, and page navigation.



3/12/26

Site running on VM IP

Nginx Reverse Proxy
-------------------

Nginx is responsible for handling all incoming web traffic. It acts as a gateway between the internet and the backend services.

Its main roles include:

  receiving HTTP requests from users

  serving frontend static files

  routing API requests to the backend server

  applying rate limiting

  adding security headers

  managing HTTPS once certificates are installed

  Nginx runs on the public web ports:

    80 (HTTP)
    443 (HTTPS)

Security Features
-----------------


Reverse Proxy Isolation

All external traffic passes through Nginx before reaching backend services. This prevents direct access to internal services and centralizes request handling.


Internal Service Ports

  Backend services run on internal ports and are not exposed to the public internet.

    Express API:
    127.0.0.1:3001

    FastAPI service:
    127.0.0.1:8000

Only the reverse proxy can communicate with them.


Security Headers

  Nginx includes several HTTP headers that improve browser security.

  These headers help prevent:

    clickjacking
    MIME type sniffing
    some cross-site scripting attacks


API Rate Limiting

  Rate limiting is enabled in Nginx to prevent excessive requests from a single IP address.

  Current limits allow approximately:

    10 requests per second per IP with a small burst allowance.


  This helps protect against:

    brute force attempts

    automated scraping

    basic denial-of-service attacks


Domain Configuration
--------------------

The domain is currently managed through Bluehost. DNS records are being updated to point the domain to the Google Cloud VM’s public IP.

Once propagation completes, users will be able to access the application through the domain instead of the server IP.



Next Steps After DNS Propagation
--------------------------------

Once the domain successfully resolves to the server, the next step is to enable HTTPS.

SSL certificates will be installed using Let’s Encrypt and Certbot. This will allow encrypted connections between users and the server.

  After HTTPS is enabled:

    HTTP traffic will redirect to HTTPS
    
    user data will be encrypted in transit

    browsers will recognize the site as secure
