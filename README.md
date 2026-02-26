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

***Response shape needs to be changed***



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

-Change expected response fields
