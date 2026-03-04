<<<<<<< HEAD
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
=======
import { createHmac, randomBytes, scryptSync, timingSafeEqual } from 'node:crypto';

const TOKEN_VERSION = 'v1';
const DEFAULT_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7;

const getSecret = () => {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_SECRET is not configured.');
  }
  return secret;
};

export const getTokenTtlSeconds = () => {
  const configured = process.env.JWT_EXPIRES_IN?.trim();
  if (!configured) return DEFAULT_TOKEN_TTL_SECONDS;

  const match = configured.match(/^(\d+)([smhd])?$/i);
  if (!match) return DEFAULT_TOKEN_TTL_SECONDS;

  const amount = Number(match[1]);
  const unit = (match[2] || 's').toLowerCase();
  const multipliers: Record<string, number> = {
    s: 1,
    m: 60,
    h: 60 * 60,
    d: 60 * 60 * 24,
  };

  return amount * (multipliers[unit] ?? 1);
};

const base64urlEncode = (value: string) => Buffer.from(value, 'utf8').toString('base64url');
const base64urlDecode = (value: string) => Buffer.from(value, 'base64url').toString('utf8');

export type AuthTokenPayload = {
  sub: string;
  email: string;
  name: string | null;
  iat: number;
  exp: number;
};

export const createAuthToken = (payload: Omit<AuthTokenPayload, 'iat' | 'exp'>) => {
  const now = Math.floor(Date.now() / 1000);
  const expiresIn = getTokenTtlSeconds();
  const body: AuthTokenPayload = {
    ...payload,
    iat: now,
    exp: now + expiresIn,
  };

  const encodedPayload = base64urlEncode(JSON.stringify(body));
  const data = `${TOKEN_VERSION}.${encodedPayload}`;
  const signature = createHmac('sha256', getSecret()).update(data).digest('base64url');
  return `${data}.${signature}`;
};

export const verifyAuthToken = (token: string): AuthTokenPayload | null => {
  const parts = token.split('.');
  if (parts.length !== 3) return null;

  const [version, encodedPayload, signature] = parts;
  if (version !== TOKEN_VERSION) return null;

  const data = `${version}.${encodedPayload}`;
  const expectedSignature = createHmac('sha256', getSecret()).update(data).digest();
  const incomingSignature = Buffer.from(signature, 'base64url');
  if (expectedSignature.length !== incomingSignature.length) return null;

  if (!timingSafeEqual(expectedSignature, incomingSignature)) {
    return null;
  }

  const payload = JSON.parse(base64urlDecode(encodedPayload)) as AuthTokenPayload;
  if (payload.exp <= Math.floor(Date.now() / 1000)) {
    return null;
  }

  return payload;
};

export const hashPassword = (password: string) => {
  const salt = randomBytes(16).toString('hex');
  const derived = scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${derived}`;
};

export const verifyPassword = (password: string, storedHash: string) => {
  const [salt, hash] = storedHash.split(':');
  if (!salt || !hash) return false;

  const derived = scryptSync(password, salt, 64);
  const stored = Buffer.from(hash, 'hex');

  if (derived.length !== stored.length) {
    return false;
  }

  return timingSafeEqual(derived, stored);
};
>>>>>>> 8e3588c (Frontend injection)
