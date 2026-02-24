import "../env"
import express, {
  Router,
  type NextFunction,
  type Request,
  type Response,
  type Express,
} from "express"
import multer from "multer"
import { randomUUID } from "node:crypto"
import { ACTIVE_CNN_MODEL, scoreWithCNN } from "../cnnScorer"
import type { AiResult } from "../cnnScorer"

const router = Router()

const JSON_CONTENT_TYPE = "application/json; charset=utf-8"
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || 15)
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

const ACCEPTED_MIMES = /^(image\/(png|jpe?g|webp|gif|heic|heif|tiff?)|application\/pdf)$/i
const ACCEPTED_EXTENSIONS = /\.(png|jpe?g|jpg|webp|gif|heic|heif|tiff?|pdf)$/i

type StoredAnalysis = {
  analysisId: string
  requestId: string
  createdAt: number
  file: { name: string; mimetype: string; size: number }
  branch: "image" | "pdf"
  verificationCode: string
  result: AiResult & {
    elapsedMs: number
  }
}

type ErrorResponsePayload = {
  error: string
  message?: string
  details?: unknown
  hint?: string
  status?: number
  requestId?: string
}

const ANALYSIS_TTL_MS = 30 * 60 * 1000
const analysisStore = new Map<string, StoredAnalysis>()

setInterval(() => {
  const now = Date.now()
  for (const [key, record] of analysisStore.entries()) {
    if (now - record.createdAt > ANALYSIS_TTL_MS) {
      analysisStore.delete(key)
    }
  }
}, ANALYSIS_TTL_MS).unref()

router.use(express.json({ limit: "2mb" }))

router.use((_req, res, next) => {
  res.type(JSON_CONTENT_TYPE)
  next()
})

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
  fileFilter: (req, file, cb) => {
    const mimetype = file.mimetype?.toLowerCase() ?? ""
    const originalName = file.originalname ?? ""
    const isAcceptedMime = ACCEPTED_MIMES.test(mimetype)
    const isFallbackMime =
      mimetype === "" ||
      mimetype === "application/octet-stream" ||
      mimetype === "application/force-download"
    const isAcceptedExt = ACCEPTED_EXTENSIONS.test(originalName)

    ;(req as any).__uploadDebug = {
      filename: originalName || null,
      mimetype: mimetype || null,
    }

    if (isAcceptedMime || (isFallbackMime && isAcceptedExt)) {
      cb(null, true)
      return
    }
    cb(new Error("Unsupported file type"))
  },
})

function handleUpload(req: Request, res: Response, next: NextFunction) {
  upload.single("file")(req, res, (err: unknown) => {
    if (!err) {
      next()
      return
    }

    if (typeof err === "object" && err && "code" in err && (err as any).code === "LIMIT_FILE_SIZE") {
      sendJsonError(res, 413, {
        error: "File too large",
        message: `Upload exceeds limit of ${MAX_UPLOAD_MB}MB`,
        details: { limitMb: MAX_UPLOAD_MB },
      })
      return
    }

    const details = err instanceof Error ? err.message : String(err)
    sendJsonError(res, 400, {
      error: "Upload error",
      message: details,
      details: {
        reason: details,
        debug: buildUploadDebug(req, (req as any).file ?? null),
      },
    })
  })
}

type FallbackResponseOptions = {
  res: Response
  requestId: string
  timestamp: string
  startedAt: number
  branch: "image" | "pdf"
  fileInfo: ReturnType<typeof serialiseFile>
  reason: string
  suggestions?: string[]
  reasons?: string[]
  provider?: AiResult["provider"]
  model?: string
  flags?: string[]
  score?: number
  logLevel?: "info" | "warn" | "error"
  logReason: string
}

function respondWithFallback(options: FallbackResponseOptions) {
  const {
    res,
    requestId,
    timestamp,
    startedAt,
    branch,
    fileInfo,
    reason,
    suggestions = [],
    reasons,
    provider = "fallback",
    model = "disabled",
    flags = ["fallback"],
    score = 50,
    logLevel = "info",
    logReason,
  } = options

  const elapsedMs = Date.now() - startedAt
  const analysisId = requestId
  const verificationCode = requestId.replace(/[^a-zA-Z0-9]/g, "").slice(0, 12).toLowerCase()

  const fallbackResult = normalizeResult({
    score,
    reasons: combineUniqueStrings([reason], reasons ?? []),
    suggestions: combineUniqueStrings(suggestions),
    flags: combineUniqueStrings(flags, ["fallback"]),
    provider,
    model,
  })

  analysisStore.set(analysisId, {
    analysisId,
    requestId,
    createdAt: Date.now(),
    branch,
    verificationCode,
    file: fileInfo,
    result: {
      ...fallbackResult,
      elapsedMs,
    },
  })

  logScoreEvent(logLevel, {
    reqId: requestId,
    timestamp,
    analysisId,
    branch,
    model: fallbackResult.model,
    file: fileInfo,
    elapsedMs,
    provider: fallbackResult.provider,
    reason: logReason,
  })

  return res.json({
    ...fallbackResult,
    requestId,
    analysisId,
    elapsedMs,
    verificationCode,
  })
}

router.get("/health", (_req: Request, res: Response) => {
  res.json({
    ok: true,
    node: process.version,
    model: ACTIVE_CNN_MODEL,
    provider: "cnn",
    fastapiUrl: process.env.FASTAPI_URL || "http://localhost:8000",
  })
})

router.post("/echo", handleUpload, (req: Request, res: Response) => {
  const uploadFile = req.file as Express.Multer.File | undefined
  if (!uploadFile) {
    return sendJsonError(res, 400, {
      error: "No file received",
      message: "Expecting multipart/form-data with single field named 'file'",
      details: buildUploadDebug(req, uploadFile),
    })
  }

  return res.json({
    name: uploadFile.originalname || "upload",
    mimetype: uploadFile.mimetype || "application/octet-stream",
    size: uploadFile.size || 0,
  })
})

const scoreRateLimiter = createRateLimiter({ windowMs: 60_000, max: 10, name: "physical-score" })

router.post("/score", scoreRateLimiter, handleUpload, async (req: Request, res: Response) => {
  const requestId = randomUUID()
  res.setHeader("x-request-id", requestId)

  const uploadFile = req.file as Express.Multer.File | undefined
  const startedAt = Date.now()
  const branch = uploadFile && isPdfFile(uploadFile) ? "pdf" : "image"
  const fileInfo = serialiseFile(uploadFile)
  const timestamp = new Date().toISOString()

  if (!uploadFile) {
    const elapsedMs = Date.now() - startedAt
    logScoreEvent("error", {
      reqId: requestId,
      timestamp,
      branch,
      model: ACTIVE_CNN_MODEL,
      file: fileInfo,
      elapsedMs,
      providerStatus: 400,
      message: "No file received",
    })
    return sendJsonError(res, 400, {
      error: "No file received",
      message: "Expecting multipart/form-data with single field named 'file'",
      details: buildUploadDebug(req, uploadFile),
      requestId,
    })
  }
const scorerEnabled = String(process.env.USE_CNN_SCORER || "true").toLowerCase() === "true"
  if (!scorerEnabled) {
    const heuristic = buildHeuristicAssessment(uploadFile, { includeFallbackNotice: false })
    return respondWithFallback({
      res,
      requestId,
      timestamp,
      startedAt,
      branch,
      fileInfo,
      reason: "CNN scoring disabled (feature flag)",
      reasons: heuristic.reasons,
      suggestions: combineUniqueStrings(heuristic.suggestions, ["Set USE_CNN_SCORER=true"]),
      flags: combineUniqueStrings(heuristic.flags, ["feature-flag-disabled"]),
      score: heuristic.score,
      provider: "heuristic",
      logReason: "feature-flag-disabled",
      logLevel: "info",
    })
  }

  try {
    const result = await scoreWithCNN({
      buffer: uploadFile.buffer,
      mimetype: uploadFile.mimetype || "application/octet-stream",
      filename: uploadFile.originalname || "upload",
    })
    const normalizedResult = normalizeResult(result)
    const elapsedMs = Date.now() - startedAt
    const analysisId = requestId
    const verificationCode = requestId.replace(/[^a-zA-Z0-9]/g, "").slice(0, 12).toLowerCase()

    analysisStore.set(analysisId, {
      analysisId,
      requestId,
      createdAt: Date.now(),
      branch,
      verificationCode,
      file: fileInfo,
      result: {
        ...normalizedResult,
        elapsedMs,
      },
    })

    logScoreEvent("info", {
      reqId: requestId,
      timestamp,
      analysisId,
      branch,
      model: normalizedResult.model,
      file: fileInfo,
      elapsedMs,
      provider: normalizedResult.provider,
    })

    return res.json({
      ...normalizedResult,
      requestId,
      analysisId,
      elapsedMs,
      verificationCode,
    })
  } catch (error: any) {
    const elapsedMs = Date.now() - startedAt
    const providerStatus = Number.isFinite(Number(error?.status)) ? Number(error.status) : undefined
    const message = error?.message ? String(error.message) : "Scoring failed"

    logScoreEvent("error", {
      reqId: requestId,
      timestamp,
      analysisId: requestId,
      branch,
      model: ACTIVE_CNN_MODEL,
      file: fileInfo,
      elapsedMs,
      providerStatus: providerStatus ?? 502,
      message,
    })

    const heuristic = buildHeuristicAssessment(uploadFile, { includeFallbackNotice: true })
    return respondWithFallback({
      res,
      requestId,
      timestamp,
      startedAt,
      branch,
      fileInfo,
      reason: "CNN scoring failed; returning local heuristic assessment.",
      reasons: heuristic.reasons,
      suggestions: combineUniqueStrings(heuristic.suggestions, [
        "Retry scoring once the CNN backend is healthy.",
        "Ensure FastAPI is running at the configured FASTAPI_URL.",
      ]),
      provider: "heuristic",
      model: ACTIVE_CNN_MODEL,
      flags: combineUniqueStrings(heuristic.flags, ["cnn-error"]),
      score: heuristic.score,
      logLevel: "error",
      logReason: "cnn-error",
    })
  }
})

//Report generation removed:CNN backend does not produce textual analysis for PDF reports.

function safelyStringify(value: unknown): string {
  if (value === undefined || value === null) return ""
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function serialiseFile(file?: Express.Multer.File | null) {
  return {
    name: file?.originalname || "upload",
    mimetype: file?.mimetype || "application/octet-stream",
    size: file?.size || 0,
  }
}

function isPdfFile(file: Express.Multer.File) {
  const mimetype = file?.mimetype?.toLowerCase() ?? ""
  return mimetype.includes("pdf") || /\.pdf$/i.test(file?.originalname || "")
}

function clampScoreValue(value: unknown): number {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 0
  return Math.min(100, Math.max(0, numeric))
}

function clampConfidenceValue(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return undefined
  return Math.min(1, Math.max(0, numeric))
}

type HeuristicAssessment = {
  score: number
  reasons: string[]
  suggestions: string[]
  flags: string[]
}

type HeuristicAssessmentOptions = {
  includeFallbackNotice?: boolean
}
function buildHeuristicAssessment(
  file?: Express.Multer.File | null,
  options: HeuristicAssessmentOptions = {}
): HeuristicAssessment {
  const includeFallbackNotice = options.includeFallbackNotice ?? true

  const reasons: string[] = []
  if (includeFallbackNotice) {
    reasons.push("Local heuristic evaluation because the CNN scorer is unavailable.")
  }
  const suggestions = new Set<string>(["Capture a clear photo or scan in good lighting."])
  const flags = new Set<string>(["local-heuristic"])

  if (!file || !file.size) {
    reasons.push("No document bytes were received to inspect.")
    suggestions.add("Upload a fresh document photo or PDF and try again.")
    flags.add("missing-upload")
    return {
      score: clampScoreValue(38),
      reasons: combineUniqueStrings(reasons),
      suggestions: combineUniqueStrings(Array.from(suggestions)),
      flags: combineUniqueStrings(Array.from(flags)),
    }
  }

  const branch = isPdfFile(file) ? "pdf" : "image"
  const sizeKb = file.size / 1024
  const mimetype = (file.mimetype || "").toLowerCase()
  const name = (file.originalname || "").toLowerCase()

  let score = branch === "pdf" ? 68 : 62
  reasons.push(`Detected ${branch.toUpperCase()} upload (~${Math.max(1, Math.round(sizeKb))}kB).`)

  if (sizeKb < 80) {
    score -= 22
    reasons.push("File is extremely small, suggesting heavy compression or a cropped screenshot.")
    suggestions.add("Increase the resolution or avoid screenshots of digital displays.")
    flags.add("low-detail")
  } else if (sizeKb < 160) {
    score -= 12
    reasons.push("File size indicates a lightweight capture that may miss paper texture cues.")
    suggestions.add("Retake the photo closer to the document to capture more detail.")
  } else if (sizeKb > 1200) {
    score += 12
    reasons.push("Large file size implies a high resolution capture with richer texture information.")
    flags.add("high-resolution")
  } else if (sizeKb > 600) {
    score += 6
    reasons.push("Document size suggests a detailed capture which often indicates an original.")
  }

  if (branch === "image") {
    if (/heic|tiff/.test(mimetype)) {
      score += 6
      reasons.push("High fidelity image format commonly produced by modern phone cameras.")
    } else if (/png/.test(mimetype) && sizeKb < 250) {
      score -= 10
      reasons.push("Small PNG images frequently originate from screenshots instead of camera captures.")
      suggestions.add("Photograph the printed document instead of sharing a digital screenshot.")
      flags.add("possible-reprint")
    }

    if (/jpeg/.test(mimetype) && sizeKb < 120) {
      score -= 8
      reasons.push("Highly compressed JPEG may hide authentic texture details.")
      suggestions.add("Use the highest quality capture settings available.")
    }
  } else {
    if (sizeKb < 150) {
      score -= 14
      reasons.push("Very small PDF files often come from exported digital documents rather than scans.")
      suggestions.add("Scan the physical copy at 300 DPI or higher for better fidelity.")
      flags.add("possible-digital-origin")
    } else if (sizeKb > 2000) {
      score += 8
      reasons.push("Large PDF indicates a detailed scan, which aligns with authentic physical paperwork.")
    }
  }

  if (/screenshot|screen[-_ ]?shot/.test(name)) {
    score -= 18
    reasons.push("Filename suggests a screenshot, increasing the likelihood of a reprint or digital capture.")
    suggestions.add("Provide a camera photo of the physical document instead of a screen capture.")
    flags.add("filename-screenshot")
  }

  if (/scan/.test(name) && branch === "pdf") {
    score += 4
    reasons.push("Filename implies a scanned document, which typically preserves physical traits.")
  }

  score = clampScoreValue(Math.round(score))

  const reasonList = reasons.length ? reasons : ["Local heuristic evaluation applied."]
  const suggestionList = combineUniqueStrings(Array.from(suggestions))
  const flagList = combineUniqueStrings(Array.from(flags))

  return {
    score,
    reasons: reasonList,
    suggestions: suggestionList,
    flags: flagList,
  }
}

function sanitizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []

  const seen = new Set<string>()
  const output: string[] = []

  const pushUnique = (raw: string | null | undefined) => {
    if (!raw) return
    const trimmed = raw.trim()
    if (!trimmed) return
    if (seen.has(trimmed)) return
    seen.add(trimmed)
    output.push(trimmed)
  }

  const extractFromEntry = (entry: unknown) => {
    if (!entry) return

    if (typeof entry === "string") {
      pushUnique(entry)
      return
    }

    if (Array.isArray(entry)) {
      for (const nested of entry) {
        extractFromEntry(nested)
      }
      return
    }

    if (typeof entry === "object") {
      const candidateKeys = [
        "text",
        "message",
        "reason",
        "label",
        "description",
        "summary",
        "value",
        "detail",
      ] as const

      let matched = false
      for (const key of candidateKeys) {
        if (key in (entry as Record<string, unknown>)) {
          matched = true
          extractFromEntry((entry as Record<string, unknown>)[key])
        }
      }

      if (!matched) {
        try {
          const stringValue = (entry as { toString?: () => string }).toString?.()
          if (typeof stringValue === "string" && stringValue !== "[object Object]") {
            pushUnique(stringValue)
          }
        } catch {
          // ignore non-serialisable objects
        }
      }

      return
    }

    if (typeof entry === "number" || typeof entry === "boolean") {
      pushUnique(String(entry))
    }
  }

  for (const entry of value) {
    extractFromEntry(entry)
  }

  return output
}

function sanitizeSubscoresRecord(value: unknown): Record<string, number> | undefined {
  if (!value || typeof value !== "object") return undefined
  const output: Record<string, number> = {}
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    const numeric = Number(raw)
    if (Number.isFinite(numeric)) {
      output[key] = clampScoreValue(numeric)
    }
  }
  return Object.keys(output).length ? output : undefined
}

function normalizeResult(result: AiResult): AiResult {
  const normalized: AiResult = {
    ...result,
    score: clampScoreValue(result.score),
    reasons: sanitizeStringArray(result.reasons),
    flags: sanitizeStringArray(result.flags),
    suggestions: sanitizeStringArray(result.suggestions),
  }

  const subscores = sanitizeSubscoresRecord(result.subscores)
  if (subscores) {
    normalized.subscores = subscores
  } else {
    delete normalized.subscores
  }

  const confidence = clampConfidenceValue(result.confidence)
  if (confidence !== undefined) {
    normalized.confidence = confidence
  } else {
    delete normalized.confidence
  }

  return normalized
}
function sendJsonError(res: Response, statusCode: number, payload: ErrorResponsePayload) {
  const responseBody: Record<string, unknown> = {
    error: payload.error || "Error",
    status: statusCode,
  }

  if (payload.message) {
    responseBody.message = payload.message
  }

  if (payload.hint) {
    responseBody.hint = payload.hint
  }

  if (payload.details !== undefined) {
    const detailString = safelyStringify(payload.details)
    if (detailString) {
      responseBody.details = detailString
    }
  }

  if (payload.requestId) {
    responseBody.requestId = payload.requestId
  }

  return res.status(statusCode).json(responseBody)
}

function buildUploadDebug(req: Request, file?: Express.Multer.File | null) {
  const rejectedFile = (req as any).__uploadDebug as
    | { filename?: string | null; mimetype?: string | null }
    | undefined

  return {
    contentType: req.headers["content-type"] || null,
    bodyKeys: Object.keys(req.body || {}),
    filename: file?.originalname || rejectedFile?.filename || null,
    mimetype: file?.mimetype || rejectedFile?.mimetype || null,
  }
}

function logScoreEvent(
  level: "info" | "error" | "warn",
  payload: Record<string, unknown>
) {
  const line = JSON.stringify({ event: "physical.score", level, ...payload })
  if (level === "error") {
    console.error(line)
    return
  }
  if (level === "warn") {
    console.warn(line)
    return
  }
  console.info(line)
}

function combineUniqueStrings(
  ...groups: Array<Iterable<string | undefined | null>>
): string[] {
  const seen = new Set<string>()
  for (const group of groups) {
    if (!group) continue
    for (const raw of group) {
      if (typeof raw !== "string") continue
      const value = raw.trim()
      if (value) {
        seen.add(value)
      }
    }
  }
  return Array.from(seen)
}

type RateLimiterOptions = {
  windowMs: number
  max: number
  name: string
}

function createRateLimiter(options: RateLimiterOptions) {
  const hits = new Map<string, number[]>()
  const windowMs = options.windowMs
  const max = options.max

  return function rateLimiter(req: Request, res: Response, next: NextFunction) {
    const key = req.ip || req.headers["x-forwarded-for"]?.toString() || "anonymous"
    const now = Date.now()
    const bucket = hits.get(key) ?? []
    const recent = bucket.filter((timestamp) => now - timestamp < windowMs)

    if (recent.length >= max) {
      console.warn(
        JSON.stringify({
          event: `${options.name}.ratelimit`,
          level: "warn",
          ip: key,
          windowMs,
          max,
        })
      )
      return sendJsonError(res, 429, {
        error: "Too many requests",
        message: "Please wait a moment before trying again.",
      })
    }

    recent.push(now)
    hits.set(key, recent)
    next()
  }
}

export default router
