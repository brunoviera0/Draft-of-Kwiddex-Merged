# Cloud Run deployment guide (`kwiddex-api`)

This guide deploys `kwiddex/server` to Cloud Run from source and connects it to Cloud SQL (MySQL) over the Cloud Run Unix socket.

## 1) Set project and APIs

```bash
gcloud config set project kwiddex-prod
gcloud services enable run.googleapis.com sqladmin.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com
```

## 2) Deploy from source

From `kwiddex/server`:

```bash
gcloud run deploy kwiddex-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

## 3) Attach Cloud SQL instance to Cloud Run service

```bash
gcloud run services update kwiddex-api \
  --region us-central1 \
  --add-cloudsql-instances kwiddex-prod:us-central1:kwiddex-mysql
```

## 4) Grant Cloud SQL Client role to service account

```bash
gcloud projects add-iam-policy-binding kwiddex-prod \
  --member="serviceAccount:kwiddex-api-sa@kwiddex-prod.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
```

If your service is using the default compute service account, bind that identity instead.

## 5) Configure Cloud Run env vars and secrets

Required app env vars:

- `BASE_PATH=""`
- `CORS_ORIGINS="https://www.kwiddex.com,https://kwiddex.com"`
- `DB_USER`, `DB_PASS`, `DB_NAME`, `INSTANCE_CONNECTION_NAME` (from Secret Manager)

Set non-secret env vars:

```bash
gcloud run services update kwiddex-api \
  --region us-central1 \
  --set-env-vars BASE_PATH=,CORS_ORIGINS=https://www.kwiddex.com\,https://kwiddex.com
```

Set secrets as env vars:

```bash
gcloud run services update kwiddex-api \
  --region us-central1 \
  --update-secrets DB_USER=DB_USER:latest,DB_PASS=DB_PASS:latest,DB_NAME=DB_NAME:latest,INSTANCE_CONNECTION_NAME=INSTANCE_CONNECTION_NAME:latest
```

You can do the same in Console:
- Cloud Run → `kwiddex-api` → Edit & Deploy New Revision → Variables & Secrets.

## 6) Verify endpoints

```bash
curl https://api.kwiddex.com/health
curl -X POST https://api.kwiddex.com/auth/signup -H 'content-type: application/json' -d '{"email":"test@example.com","password":"StrongPass123!"}'
```

## 7) Map custom domain (`api.kwiddex.com`)

Create mapping:

```bash
gcloud beta run domain-mappings create \
  --service kwiddex-api \
  --domain api.kwiddex.com \
  --region us-central1
```

Describe mapping and copy DNS records:

```bash
gcloud beta run domain-mappings describe \
  --domain api.kwiddex.com \
  --region us-central1
```

Then create the shown DNS records at your DNS provider and wait for TLS provisioning.
