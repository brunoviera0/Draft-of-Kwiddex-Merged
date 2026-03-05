import { createHmac, timingSafeEqual } from "node:crypto"

const getJwtSecret = (): string => {
  const secret = process.env.KWX_JWT_SECRET
  if (!secret) {
    throw new Error(
      "KWX_JWT_SECRET is not configured. This must match the FastAPI backend's KWX_JWT_SECRET."
    )
  }
  return secret
}

export type JwtPayload = {
  sub: string       //user_id (e.g. "USR-ABC123DEF456")
  username: string   //email used as username
  iat: number
  exp: number
}


//Verify a FastAPI-issued JWT (HS256).
export const verifyFastapiToken = (token: string): JwtPayload | null => {
  try {
    const parts = token.split(".")
    if (parts.length !== 3) return null

    const [header, payload, signature] = parts
    const data = `${header}.${payload}`
    const secret = getJwtSecret()

    const expected = createHmac("sha256", secret).update(data).digest()
    const incoming = Buffer.from(signature, "base64url")

    if (expected.length !== incoming.length) return null
    if (!timingSafeEqual(expected, incoming)) return null

    const decoded = JSON.parse(
      Buffer.from(payload, "base64url").toString("utf8")
    ) as JwtPayload

    if (decoded.exp && decoded.exp <= Math.floor(Date.now() / 1000)) {
      return null
    }

    return decoded
  } catch {
    return null
  }
}

//Express middleware
export const requireAuth = (req: any, res: any, next: any) => {
  const authHeader = req.header("authorization") || ""
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : ""

  if (!token) {
    return res.status(401).json({ error: "Missing access token." })
  }

  const payload = verifyFastapiToken(token)
  if (!payload) {
    return res.status(401).json({ error: "Invalid or expired access token." })
  }

  req.user = payload
  next()
}
