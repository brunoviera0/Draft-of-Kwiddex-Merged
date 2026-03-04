# Bluehost database setup (what you need to do)

This guide prepares a production-ready MySQL database for KwiddeX on Bluehost.

## 1) Create the MySQL database in Bluehost cPanel
1. Open **cPanel**.
2. Go to **MySQL Databases**.
3. Create a new database (example: `kwiddex`).
4. Create a database user (example: `kwiddex_user`) with a strong password.
5. Add that user to the database with **ALL PRIVILEGES**.

> Bluehost usually prefixes names (e.g. `accountname_kwiddex`). Use the exact final names shown in cPanel.

## 2) Import the production schema
1. Open **phpMyAdmin** from cPanel.
2. Select your database.
3. Open the **Import** tab.
4. Import this file from the repo:
   - `server/sql/bluehost-auth-schema.sql`
5. Confirm tables were created (users, organizations, projects, documents, audit_logs, etc).

## 3) Configure server environment variables
Set these on your server (`server/.env.local` or host environment config):

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_full_db_name
DB_USER=your_full_db_user
DB_PASSWORD=your_db_password
JWT_SECRET=generate-a-very-long-random-secret
JWT_EXPIRES_IN=7d
```

Also keep existing app variables (`OPENAI_API_KEY`, `OPENAI_MODEL`, etc.) configured.

## 4) Security checklist (recommended before launch)
- Use unique, long DB and JWT secrets.
- Restrict DB user to this app database only.
- Enable automatic backups in Bluehost.
- Keep `utf8mb4` collation for full Unicode support.
- Store only hashed passwords/tokens in app logic.
- Enforce HTTPS in production.

## 5) Ongoing operations
- For schema changes, create versioned SQL migration files (example: `server/sql/migrations/2026-02-16-add-foo.sql`).
- Run migrations first in staging, then production.
- Add monitoring for slow queries and failed connections.

## Included database foundation
The schema is intentionally broader than login/signup so you can scale without redesigning core tables:
- **Identity/auth**: `users`, `roles`, `user_roles`, `login_sessions`, password/email token tables.
- **Multi-tenant model**: `organizations`, `organization_members`, `projects`.
- **Document workflows**: `documents`, `verification_results`.
- **Platform operations**: `api_keys`, `webhooks`, `audit_logs`.
