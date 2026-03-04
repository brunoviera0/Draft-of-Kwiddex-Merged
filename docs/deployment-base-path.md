# Deployment note: cPanel / Passenger Application URL prefix

If your app is mounted at a URL prefix (for example cPanel **Application URL** is `/warwatch`), configure both frontend and backend with the same prefix:

- Backend env: `BASE_PATH=/warwatch`
- Frontend env: `VITE_API_BASE=/warwatch`

With this setup:

- Health endpoint: `GET /warwatch/health` → `{ "ok": true }`
- Signup endpoint: `POST /warwatch/auth/signup` → JSON response

## Where to set env vars in this repo

- Local frontend env: `kwiddex/.env.local` (or `kwiddex/.env`)
- Local API env: `kwiddex/server/.env.local` (or `kwiddex/server/.env`)
- Shared fallback env loader also reads: `kwiddex/.env.local` and `kwiddex/.env`

## Where to set env vars on cPanel

Set `BASE_PATH` in your Node.js app environment variables in cPanel (Passenger app settings).
Set `VITE_API_BASE` at frontend build time in the environment used for `npm run build`.

If your app is deployed at root (`/`), leave both values empty.
