<<<<<<< HEAD
import { Router } from "express"
import { verifyFastapiToken } from "../auth"

const FASTAPI_BASE = (process.env.FASTAPI_URL || "http://localhost:8000").replace(/\/+$/, "")
const authRouter = Router()
const sanitizeEmail = (value: unknown) => String(value ?? "").trim().toLowerCase()

authRouter.post("/signup", async (req, res) => {
  const email = sanitizeEmail(req.body?.email)
  const password = String(req.body?.password ?? "")
  const fullName = String(req.body?.fullName ?? "").trim() || undefined

  if (!email || !password) {
    return res.status(400).json({ error: "Email and password are required." })
  }

  try {
    const fastapiRes = await fetch(`${FASTAPI_BASE}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: email, password, organization: fullName || null }),
    })

    const data = await fastapiRes.json()

    if (!data.success) {
      return res.status(fastapiRes.status >= 400 ? fastapiRes.status : 400).json({
        error: data.message || "Registration failed.",
      })
    }

    //Auto-login after registration
    const loginRes = await fetch(`${FASTAPI_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: email, password }),
    })
    const loginData = await loginRes.json()

    if (!loginData.success || !loginData.token) {
      return res.status(201).json({ ok: true })
    }

    return res.status(201).json({
      token: loginData.token,
      user: { id: loginData.user_id, email },
    })
  } catch (error: any) {
    console.error("[auth] Signup proxy failed:", error?.message)
    return res.status(502).json({ error: "Unable to reach authentication service." })
  }
})

authRouter.post("/login", async (req, res) => {
  const email = sanitizeEmail(req.body?.email)
  const password = String(req.body?.password ?? "")

  if (!email || !password) {
    return res.status(400).json({ error: "Email and password are required." })
  }

  try {
    const fastapiRes = await fetch(`${FASTAPI_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: email, password }),
    })
    const data = await fastapiRes.json()

    if (!data.success || !data.token) {
      return res.status(401).json({ error: data.message || "Invalid email or password." })
    }

    return res.json({
      token: data.token,
      user: { id: data.user_id, email },
    })
  } catch (error: any) {
    console.error("[auth] Login proxy failed:", error?.message)
    return res.status(502).json({ error: "Unable to reach authentication service." })
  }
})

authRouter.get("/me", (req, res) => {
  const authHeader = req.header("authorization") || ""
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : ""

  if (!token) return res.status(401).json({ error: "Missing access token." })

  const payload = verifyFastapiToken(token)
  if (!payload) return res.status(401).json({ error: "Invalid or expired access token." })

  return res.json({ user: { id: payload.sub, email: payload.username } })
})

export default authRouter
=======
import bcrypt from 'bcrypt';
import { Router } from 'express';
import { createAuthToken, verifyAuthToken } from '../auth';
import { ensureUsersTable, getDbPool } from '../db';

type UserRow = {
  id: number;
  email: string;
  password_hash: string;
};

const authRouter = Router();

const sanitizeEmail = (value: unknown) => String(value ?? '').trim().toLowerCase();

const publicUser = (user: { id: number; email: string }) => ({
  id: String(user.id),
  email: user.email,
});

const getUserByEmail = async (email: string): Promise<UserRow | null> => {
  await ensureUsersTable();

  const [rows] = await getDbPool().execute<UserRow[]>(
    'SELECT id, email, password_hash FROM users WHERE email = ? LIMIT 1;',
    [email],
  );

  return rows[0] ?? null;
};

authRouter.post('/signup', async (req, res) => {
  const email = sanitizeEmail(req.body?.email);
  const password = String(req.body?.password ?? '');

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required.' });
  }

  try {
    await ensureUsersTable();

    const passwordHash = await bcrypt.hash(password, 10);
    await getDbPool().execute('INSERT INTO users (email, password_hash) VALUES (?, ?);', [email, passwordHash]);

    return res.status(201).json({ ok: true });
  } catch (error: any) {
    if (error?.code === 'ER_DUP_ENTRY') {
      return res.status(409).json({ error: 'Email already exists.' });
    }

    console.error('Signup failed:', error);
    return res.status(500).json({ error: 'Unable to create account right now.' });
  }
});

authRouter.post('/login', async (req, res) => {
  const email = sanitizeEmail(req.body?.email);
  const password = String(req.body?.password ?? '');

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required.' });
  }

  try {
    const user = await getUserByEmail(email);
    if (!user) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const matches = await bcrypt.compare(password, user.password_hash);
    if (!matches) {
      return res.status(401).json({ error: 'Invalid email or password.' });
    }

    const token = createAuthToken({
      sub: String(user.id),
      email: user.email,
      name: null,
    });

    return res.json({
      token,
      user: publicUser(user),
    });
  } catch (error) {
    console.error('Login failed:', error);
    return res.status(500).json({ error: 'Unable to log in right now.' });
  }
});

authRouter.get('/me', async (req, res) => {
  const authHeader = req.header('authorization') || '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';

  if (!token) {
    return res.status(401).json({ error: 'Missing access token.' });
  }

  try {
    const payload = verifyAuthToken(token);
    if (!payload) {
      return res.status(401).json({ error: 'Invalid or expired access token.' });
    }

    const user = await getUserByEmail(payload.email);
    if (!user) {
      return res.status(401).json({ error: 'User no longer exists.' });
    }

    return res.json({ user: publicUser(user) });
  } catch (error) {
    console.error('Auth check failed:', error);
    return res.status(500).json({ error: 'Unable to validate session.' });
  }
});

export default authRouter;
>>>>>>> 8e3588c (Frontend injection)
