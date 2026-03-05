The purpose of this repository is to test the integration of Kwiddex. From the following repositories:

CNN Model/FastAPI backend: https://github.com/brunoviera0/Kwiddex-CNN-Model

Express Server/Frontend: https://github.com/Kwiddex/kwiddex



## Architecture

```
Browser → Vite (React) :5173
             ↓ /api/*
         Express BFF :3001
             ↓
         FastAPI CNN Backend :8000
         (PyTorch ResNet18, Monte Carlo inference)
```

## Project Structure

```
├── backend/          FastAPI — CNN model, certification, RSA signing
│   ├── predict.py        Main API (predict, monte_carlo, certify, verify)
│   ├── certificate_store.py  Certificate persistence
│   └── best_real_fake_resnet18.pt  Trained model weights
├── server/           Express BFF - proxies auth + scoring to FastAPI
│   └── src/
│       ├── index.ts          Server entry
│       ├── cnnScorer.ts      FastAPI client (CnnResult type)
│       ├── routes/
│       │   ├── physical.ts   /physical/score endpoint
│       │   └── auth.ts       /auth/* proxy to FastAPI
│       └── env.ts            Environment config
├── frontend/         React (Vite) — 6-page SPA
│   ├── src/pages/
│   │   ├── Physical.jsx      CNN analysis tool (home page)
│   │   ├── Verify.jsx        Certificate verification
│   │   ├── Ocr.jsx           PDF text extraction
│   │   ├── Sign.jsx          Document certification (auth required)
│   │   ├── Auth.jsx          Login / signup
│   │   └── About.jsx         Project info
│   └── cypress/              E2E tests (navigation, auth, scoring)
└── tests/            Backend integration tests (bash)
    └── test_express_fastapi.sh   38 assertions across 7 groups
```

## Pages

| Route | Auth | Description |
|-------|------|-------------|
| `/` | No | Upload document → CNN confidence %, 95% CI bounds, Monte Carlo stats |
| `/verify` | No | Upload signed PDF → verify Kwiddex certificate |
| `/ocr` | No | Upload PDF → extract text content |
| `/sign` | Yes | Upload document → create RSA-signed certified PDF |
| `/auth` | — | Login / signup |
| `/about` | — | Project information |

## CNN Response (CnnResult)

The scoring endpoint returns raw model output with no labels or interpretation:

```json
{
  "confidence": 0.8028,
  "confidenceInterval": { "lower": 0.2127, "upper": 0.9587 },
  "monteCarloStats": { "numSamples": 30, "agreementRate": 0.933, "stdDev": 0.206 },
  "provider": "cnn",
  "model": "resnet18-real-fake"
}
```

No "real/fake" labels. No score out of 100. Human review required for all decisions.

## Running Locally

```bash
# Terminal 1 — FastAPI
cd backend && uvicorn predict:app --host 0.0.0.0 --port 8000

# Terminal 2 — Express
cd server && npm install && npm run dev:api

# Terminal 3 — Frontend
cd frontend && npm install && npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173`

## Testing

```bash
# Backend integration (38 tests — health, auth, JWT, CNN scoring, PDF metadata, errors)
bash tests/test_express_fastapi.sh

# Frontend E2E (requires Cypress + display server)
cd frontend && npx cypress run

# FastAPI unit tests (56 tests)
cd backend && python -m pytest unit_tests_api.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTAPI_URL` | `http://localhost:8000` | FastAPI backend URL |
| `KWX_JWT_SECRET` | `` | Shared JWT signing secret |
| `USE_CNN_SCORER` | `true` | Enable/disable CNN scoring |
| `USE_MONTE_CARLO` | `true` | Use Monte Carlo inference |
| `MC_SAMPLES` | `30` | Number of MC augmentation samples |

## Recent Changes

**Frontend v2** — While merging frontend, simplified from 12 pages to 6. Physical analysis is the home page. Verify and OCR are public. Sign requires authentication (user ID ties to certificate). Removed: Landing, Home, Dashboard, DocumentationHub, Metadata.

**CnnResult** — Replaced AiResult interpretation layer. Express passes through raw CNN confidence and CI bounds. No labels, no score mapping, no reasons/flags/suggestions. Frontend displays numbers only.

**Certification** — Removed auto-reject based on CNN prediction. Certification is now a human-approval workflow. CNN confidence is recorded but does not gate signing.

## Next: Security (Week of 3/10)

1. No exposed API keys in code, only API calls via edge functions
2. Input validation and sanitization for all user inputs
3. Rate limiting on all API endpoints
4. Replace custom JWT auth with managed platform (Clerk, Firebase, Supabase, or Auth0 — sponsor's choice)

