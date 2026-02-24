import express from 'express';
import type { Request, Response } from 'express';
import { limit } from './emailLimiter';

const EMAIL_FROM =
  process.env.EMAIL_FROM || 'KwiddeX Intelligence <noreply@yourdomain.com>';
const RESEND_API_KEY = process.env.RESEND_API_KEY || '';
const PUBLIC_BASE_URL =
  process.env.NEXT_PUBLIC_BASE_URL || process.env.PUBLIC_BASE_URL || '';

type Payload = {
  to: string;
  subject?: string;
  signerEmail?: string;
  signatureCode?: string;
  documentId?: string;
  documentUrl?: string;
};

const router = express.Router();

router.use(express.json());

const resolveClientIp = (req: Request) => {
  const header = req.headers['x-forwarded-for'];
  if (Array.isArray(header)) {
    return header[0];
  }
  if (typeof header === 'string' && header.length > 0) {
    return header.split(',')[0]?.trim() || 'local';
  }
  return req.ip || 'local';
};

router.post('/email', async (req: Request, res: Response) => {
  const ip = resolveClientIp(req);
  if (!limit(ip)) {
    return res.status(429).json({ error: 'Rate limit exceeded' });
  }

  const body = req.body as Payload;

  if (!body?.to) {
    return res.status(400).json({ error: "Missing 'to' email." });
  }

  const to = body.to.trim().toLowerCase();

  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)) {
    return res.status(400).json({ error: 'Invalid email format.' });
  }

  const subject = body.subject || 'Your document signature code';
  const signature = body.signatureCode || '—';
  const docLink =
    body.documentUrl ||
    (body.documentId && PUBLIC_BASE_URL
      ? `${PUBLIC_BASE_URL}/digital/verify/${body.documentId}`
      : '');

  const html = `
      <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.4">
        <h2>KwiddeX Intelligence — Signature Details</h2>
        <p>Hi${body.signerEmail ? ` ${body.signerEmail}` : ''},</p>
        <p>Here is your signature code:</p>
        <p style="font-size:20px;font-weight:700;letter-spacing:0.5px">${signature}</p>
        ${
          docLink
            ? `<p>You can view or verify the document here:<br>
               <a href="${docLink}">${docLink}</a></p>`
            : ''
        }
        <hr style="margin:16px 0;opacity:.2">
        <p style="color:#666">If you didn’t request this, you can ignore this email.</p>
      </div>
    `;

  if (RESEND_API_KEY) {
    try {
      const response = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${RESEND_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          from: EMAIL_FROM,
          to,
          subject,
          html
        })
      });

      if (!response.ok) {
        const details = await response.text();
        return res.status(502).json({ error: 'Email provider error', details });
      }

      const data = await response.json();
      return res.json({ ok: true, id: data?.id ?? null });
    } catch (error) {
      return res
        .status(502)
        .json({ error: 'Email provider error', details: String(error) });
    }
  }

  console.warn('[email:fallback] RESEND_API_KEY not set — printing email to console');
  console.log({ from: EMAIL_FROM, to, subject, html });
  return res.json({ ok: true, simulated: true });
});

export { router as emailRouter };
