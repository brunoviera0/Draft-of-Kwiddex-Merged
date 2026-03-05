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
