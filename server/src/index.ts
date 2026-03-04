<<<<<<< HEAD
import "./env"
import cors from "cors"
import express from "express"
import multer from "multer"
import { PDFDocument } from "pdf-lib"
import { createHash } from "node:crypto"
import { createServer } from "node:http"
import { emailRouter } from "./email"
import physicalRouter from "./routes/physical"
import authRouter from "./routes/auth"
import wpRouter from "./routes/wordpress"

const MIN_NODE_MAJOR = 18
const nodeMajor = Number(process.versions.node.split(".")[0] || "0")
if (Number.isFinite(nodeMajor) && nodeMajor < MIN_NODE_MAJOR) {
  console.error(
    `KwiddeX API requires Node ${MIN_NODE_MAJOR}+. Detected ${process.versions.node}. Please upgrade.`
  )
  process.exit(1)
}

const normalizeBasePath = (value: string | undefined) => {
  const trimmed = String(value ?? "").trim()
  if (!trimmed || trimmed === "/") return ""
  const withLeadingSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`
  return withLeadingSlash.replace(/\/+$/, "")
}

const BASE_PATH = normalizeBasePath(process.env.BASE_PATH)

const configuredOrigins = (process.env.CORS_ORIGINS ?? "")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean)

const defaultProductionOrigins = ["https://www.kwiddex.com", "https://kwiddex.com"]
const localDevelopmentOrigins = ["http://localhost:3000", "http://localhost:5173"]

const fallbackOrigins =
  process.env.NODE_ENV === "production"
    ? defaultProductionOrigins
    : [...defaultProductionOrigins, ...localDevelopmentOrigins]

const allowedOrigins = new Set(configuredOrigins.length > 0 ? configuredOrigins : fallbackOrigins)

const corsOptions = {
  origin(origin, callback) {
    if (!origin || allowedOrigins.has(origin)) return callback(null, true)
    return callback(new Error("Origin not allowed by CORS"))
  },
  methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true,
}

const app = express()
app.set("trust proxy", 1)
=======
import './env';
import cors from 'cors';
import express from 'express';
import multer from 'multer';
import { PDFDocument } from 'pdf-lib';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { emailRouter } from './email';
import physicalRouter from './routes/physical';
import authRouter from './routes/auth';
import { checkDbHealth, closeDbPool, ensureUsersTable } from './db';

const MIN_NODE_MAJOR = 18;
const nodeMajor = Number(process.versions.node.split('.')[0] || '0');
if (Number.isFinite(nodeMajor) && nodeMajor < MIN_NODE_MAJOR) {
  console.error(
    `KwiddeX Intelligence API requires Node ${MIN_NODE_MAJOR}+ for OpenAI integrations. Detected ${process.versions.node}. Please upgrade.`,
  );
  process.exit(1);
}

const normalizeBasePath = (value: string | undefined) => {
  const trimmed = String(value ?? '').trim();
  if (!trimmed || trimmed === '/') {
    return '';
  }

  const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return withLeadingSlash.replace(/\/+$/, '');
};

const BASE_PATH = normalizeBasePath(process.env.BASE_PATH);

const configuredOrigins = (process.env.CORS_ORIGINS ?? '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

const defaultProductionOrigins = ['https://www.kwiddex.com', 'https://kwiddex.com'];
const localDevelopmentOrigins = ['http://localhost:3000', 'http://localhost:5173'];

const fallbackOrigins = process.env.NODE_ENV === 'production'
  ? defaultProductionOrigins
  : [...defaultProductionOrigins, ...localDevelopmentOrigins];

const allowedOrigins = new Set(configuredOrigins.length > 0 ? configuredOrigins : fallbackOrigins);

const corsOptions = {
  origin(origin, callback) {
    if (!origin || allowedOrigins.has(origin)) {
      return callback(null, true);
    }

    return callback(new Error('Origin not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
};

const app = express();
app.set('trust proxy', 1);
>>>>>>> 8e3588c (Frontend injection)

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
<<<<<<< HEAD
})

app.use(cors(corsOptions))
app.options("*", cors(corsOptions))
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const KNOWN_EDITORS = [
  "Photoshop", "Acrobat", "Preview", "Illustrator", "Microsoft Word",
  "Canva", "Google", "LibreOffice", "Foxit", "TinyWow",
  "Wondershare", "Nitro", "Sejda",
]
=======
});

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const KNOWN_EDITORS = [
  'Photoshop',
  'Acrobat',
  'Preview',
  'Illustrator',
  'Microsoft Word',
  'Canva',
  'Google',
  'LibreOffice',
  'Foxit',
  'TinyWow',
  'Wondershare',
  'Nitro',
  'Sejda',
];
>>>>>>> 8e3588c (Frontend injection)

const registerRoutes = (prefix: string) => {
  app.get(`${prefix}/health`, async (_req, res) => {
    try {
<<<<<<< HEAD
      return res.status(200).json({
        ok: true,
        fastapi: process.env.FASTAPI_URL || "http://localhost:8000",
        wordpress: process.env.WORDPRESS_URL || "(not configured)",
      })
    } catch (error) {
      console.error("Health check failed:", error)
      return res.status(500).json({ ok: false })
    }
  })

  app.use(`${prefix}`, emailRouter)
  app.use(`${prefix}/physical`, physicalRouter)
  app.use(`${prefix}/auth`, authRouter)
  app.use(`${prefix}/wp`, wpRouter)

  app.post(`${prefix}/verify`, upload.single("file"), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" })
    }

    try {
      const fileBuffer = req.file.buffer
      const pdf = await PDFDocument.load(fileBuffer, { ignoreEncryption: true })
      const sha256 = createHash("sha256").update(fileBuffer).digest("hex")
      const creationDate = pdf.getCreationDate()
      const modificationDate = pdf.getModificationDate()
=======
      await checkDbHealth();
      return res.status(200).json({ ok: true });
    } catch (error) {
      console.error('Health check failed:', error);
      return res.status(500).json({ ok: false });
    }
  });

  app.use(`${prefix}`, emailRouter);
  app.use(`${prefix}/physical`, physicalRouter);
  app.use(`${prefix}/auth`, authRouter);

  app.post(`${prefix}/verify`, upload.single('file'), async (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    try {
      const fileBuffer = req.file.buffer;
      const pdf = await PDFDocument.load(fileBuffer, { ignoreEncryption: true });

      const sha256 = createHash('sha256').update(fileBuffer).digest('hex');

      const creationDate = pdf.getCreationDate();
      const modificationDate = pdf.getModificationDate();
>>>>>>> 8e3588c (Frontend injection)

      const core = {
        title: pdf.getTitle() ?? null,
        author: pdf.getAuthor() ?? null,
        creator: pdf.getCreator() ?? null,
        producer: pdf.getProducer() ?? null,
        creationDate: creationDate ? creationDate.toISOString() : null,
        modDate: modificationDate ? modificationDate.toISOString() : null,
<<<<<<< HEAD
      }

      const searchableFields = [core.creator, core.producer]
        .filter((v): v is string => Boolean(v))
        .map((v) => v.toLowerCase())

      const knownEditor = KNOWN_EDITORS.find((e) =>
        searchableFields.some((f) => f.includes(e.toLowerCase()))
      )

      return res.json({ sha256, core, knownEditor: knownEditor ?? null })
    } catch (error) {
      console.error("Failed to verify PDF metadata", error)
      return res.status(400).json({
        error: "Unable to process PDF. Ensure the file is not corrupted or encrypted.",
      })
    }
  })
}

registerRoutes(BASE_PATH)
registerRoutes(`${BASE_PATH}/api`)

app.use((_req, res) => { res.status(404).json({ error: "Not found" }) })
app.use((error, _req, res, _next) => {
  console.error("Unhandled server error:", error)
  res.status(500).json({ error: "Internal server error" })
})

const PORT = Number(process.env.PORT ?? 3001)
const HOST = "0.0.0.0"
const server = createServer(app)

server.listen(PORT, HOST, () => {
  console.log(`KwiddeX API listening on ${HOST}:${PORT} (base: ${BASE_PATH || "/"})`)
  console.log(`  FastAPI: ${process.env.FASTAPI_URL || "http://localhost:8000"}`)
  console.log(`  WordPress: ${process.env.WORDPRESS_URL || "(not configured)"}`)
})

let shuttingDown = false
const shutdown = async (signal: string) => {
  if (shuttingDown) return
  shuttingDown = true
  console.log(`Received ${signal}; shutting down.`)
  server.close(() => process.exit())
  setTimeout(() => { console.error("Shutdown timed out."); process.exit(1) }, 10_000).unref()
}

process.on("SIGTERM", () => { void shutdown("SIGTERM") })
process.on("SIGINT", () => { void shutdown("SIGINT") })
=======
      };

      const searchableFields = [core.creator, core.producer]
        .filter((value): value is string => Boolean(value))
        .map((value) => value.toLowerCase());

      const knownEditor = KNOWN_EDITORS.find((editor) =>
        searchableFields.some((field) => field.includes(editor.toLowerCase())),
      );

      return res.json({
        sha256,
        core,
        knownEditor: knownEditor ?? null,
      });
    } catch (error) {
      console.error('Failed to verify PDF metadata', error);
      return res.status(400).json({ error: 'Unable to process PDF. Ensure the file is not corrupted or encrypted.' });
    }
  });
};

registerRoutes(BASE_PATH);
registerRoutes(`${BASE_PATH}/api`);

app.use((_req, res) => {
  res.status(404).json({ error: 'Not found' });
});

app.use((error, _req, res, _next) => {
  console.error('Unhandled server error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = Number(process.env.PORT ?? 8080);
const HOST = '0.0.0.0';
const server = createServer(app);

ensureUsersTable().catch((error) => {
  console.error('Failed to ensure users table exists at startup:', error);
});

server.listen(PORT, HOST, () => {
  const mountHint = BASE_PATH || '/';
  console.log(`KwiddeX Intelligence API listening on ${HOST}:${PORT} (base path: ${mountHint})`);
});

let shuttingDown = false;
const shutdown = async (signal: string) => {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log(`Received ${signal}; shutting down gracefully.`);

  server.close(async (closeError) => {
    if (closeError) {
      console.error('HTTP server close failed:', closeError);
      process.exitCode = 1;
    }

    try {
      await closeDbPool();
    } catch (dbError) {
      console.error('Database pool shutdown failed:', dbError);
      process.exitCode = 1;
    } finally {
      process.exit();
    }
  });

  setTimeout(() => {
    console.error('Graceful shutdown timed out; forcing exit.');
    process.exit(1);
  }, 10_000).unref();
};

process.on('SIGTERM', () => {
  void shutdown('SIGTERM');
});

process.on('SIGINT', () => {
  void shutdown('SIGINT');
});
>>>>>>> 8e3588c (Frontend injection)
