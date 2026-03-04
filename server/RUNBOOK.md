# Cloud Run Runbook (kwiddex-api)

## Local development

```bash
cd kwiddex/server
npm install
export DB_USER="root"
export DB_PASS="password"
export DB_NAME="kwiddex"
export DB_HOST="127.0.0.1"
export DB_PORT="3306"
npm run build
npm start
```

Health check:

```bash
curl -i http://127.0.0.1:8080/health
```

## Required environment variables

- `DB_USER` (required)
- `DB_PASS` (or `DB_PASSWORD`) (required for authenticated DB)
- `DB_NAME` (required)
- `INSTANCE_CONNECTION_NAME` (required on Cloud Run for Cloud SQL socket mode)
- `DB_HOST` + `DB_PORT` (used for local TCP fallback when `INSTANCE_CONNECTION_NAME` is not set)
- `JWT_SECRET` (recommended for stable auth tokens)
- `BASE_PATH` (optional path prefix)
- `CORS_ORIGINS` (optional comma-separated allow-list additions)

## Cloud Run deploy (build from source)

```bash
cd kwiddex/server
gcloud run deploy kwiddex-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --add-cloudsql-instances "<PROJECT:REGION:INSTANCE>" \
  --set-env-vars "INSTANCE_CONNECTION_NAME=<PROJECT:REGION:INSTANCE>,DB_USER=<DB_USER>,DB_NAME=<DB_NAME>,CORS_ORIGINS=https://kwiddex.com,https://www.kwiddex.com" \
  --set-secrets "DB_PASS=<SECRET_NAME>:latest,JWT_SECRET=<JWT_SECRET_NAME>:latest"
```

## Verify deployment

```bash
API_URL="https://api.kwiddex.com"
curl -i "${API_URL}/health"
```

If `BASE_PATH` is configured (example `/api`), verify `${API_URL}/api/health` as well.

## Domain + frontend notes

- Map Cloud Run service to `api.kwiddex.com` using Cloud Run domain mappings and DNS records.
- In Vercel, set `VITE_API_BASE=https://api.kwiddex.com` so frontend API calls never default to localhost in production.
