import "./env"
import { createHash } from "node:crypto"

export type AiResult = {
  score: number
  reasons: string[]
  flags: string[]
  suggestions: string[]
  subscores?: Record<string, number>
  confidence?: number
  provider: "openai" | "fallback" | "heuristic"
  model: string
  qualitySignals?: QualitySignals
}

type QualitySignals = {
  previewDataUrl?: string
  quality_blur?: number
  quality_glare?: number
  quality_skew?: number
  metrics?: {
    width?: number
    height?: number
  }
}

const API_BASE = "https://api.openai.com/v1"
const API_KEY = process.env.OPENAI_API_KEY?.trim() ?? ""
export const ACTIVE_OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini"
const TEMPERATURE = Number.isFinite(Number(process.env.OPENAI_TEMPERATURE))
  ? Number(process.env.OPENAI_TEMPERATURE)
  : 0.2
const OPENAI_SEED = Number.isFinite(Number(process.env.OPENAI_SEED))
  ? Number(process.env.OPENAI_SEED)
  : 1337
const OPENAI_FILE_PURPOSE = process.env.OPENAI_FILE_PURPOSE?.trim() || "vision"
const OPENAI_FILE_POLL_INTERVAL_MS = Number.isFinite(Number(process.env.OPENAI_FILE_POLL_INTERVAL_MS))
  ? Number(process.env.OPENAI_FILE_POLL_INTERVAL_MS)
  : 400
const OPENAI_FILE_POLL_TIMEOUT_MS = Number.isFinite(Number(process.env.OPENAI_FILE_POLL_TIMEOUT_MS))
  ? Number(process.env.OPENAI_FILE_POLL_TIMEOUT_MS)
  : 10_000

const SCHEMA = {
  name: "DocumentAuthenticityVerdict",
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["score", "reasons", "flags", "suggestions"],
    properties: {
      score: { type: "number", minimum: 0, maximum: 100 },
      reasons: { type: "array", items: { type: "string" }, maxItems: 8 },
      flags: { type: "array", items: { type: "string" }, maxItems: 10 },
      suggestions: { type: "array", items: { type: "string" }, maxItems: 8 },
      confidence: { type: "number", minimum: 0, maximum: 1 },
      subscores: {
        type: "object",
        additionalProperties: { type: "number", minimum: 0, maximum: 100 },
      },
    },
  },
} as const

const PROMPT = `You are a document image/scan forensics assistant.
Given a single uploaded document (image or PDF) and optional local quality metrics, estimate the likelihood it's an original (not digitally altered/reprinted).
Consider: paper texture/fibers, ink halation, resampling/blockiness, ELA-like cues, glare, blur, skew/cropping, shadows, background context.
Always map the provided local metrics into aligned subscores when possible (0-100 where higher = better quality).
Return JSON strictly matching the schema with: score 0-100 (higher = more likely original), reasons, flags, suggestions, optional subscores for quality_blur, quality_glare, quality_skew, tamper_ela, resampling, blockiness, ocr_consistency and optional confidence (0-1).
If the document is very poor quality make that clear in reasons and lower the confidence.`

type DocumentContext = {
  metadataSummary: string
  textSnippet?: string
  imageFingerprint?: string
}

type OpenAiContent = Array<{ type: string; [key: string]: unknown }>

const JSON_HEADERS = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${API_KEY}`,
}

function bufferToDataUrl(buf: Buffer, mime = "image/jpeg") {
  return `data:${mime};base64,${buf.toString("base64")}`
}

async function readErrorPayload(response: Response): Promise<string> {
  try {
    return await response.text()
  } catch (readError) {
    return readError instanceof Error ? readError.message : String(readError)
  }
}

async function uploadPdf(buffer: Buffer, filename: string) {
  const formData = new FormData()
  const safeName = filename?.trim() || "document.pdf"
  const blob = new Blob([buffer], { type: "application/pdf" })
  formData.append("purpose", OPENAI_FILE_PURPOSE)
  formData.append("file", blob, safeName.endsWith(".pdf") ? safeName : `${safeName}.pdf`)

  const response = await fetch(`${API_BASE}/files`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
    },
    body: formData,
  })

  if (!response.ok) {
    const payload = await readErrorPayload(response)
    throw Object.assign(new Error("OpenAI file upload failed"), {
      status: response.status,
      details: { endpoint: "files", status: response.status, body: payload },
    })
  }

  const payload = await response.json()
  const fileId = payload?.id as string
  if (fileId) {
    await waitForFileProcessing(fileId)
  }
  return fileId
}

async function waitForFileProcessing(fileId: string) {
  const deadline = Date.now() + OPENAI_FILE_POLL_TIMEOUT_MS
  let lastStatus = ""
  let lastError: unknown

  while (Date.now() < deadline) {
    try {
      const metadata = await fetchFileMetadata(fileId)
      const status = typeof metadata?.status === "string" ? metadata.status.toLowerCase() : ""
      lastStatus = status

      if (status === "error") {
        const reason = metadata?.last_error ?? metadata?.status_details ?? metadata
        throw Object.assign(new Error("OpenAI file processing failed"), {
          status: 502,
          details: { endpoint: "files.status", status, body: reason },
        })
      }

      if (!status || status === "processed" || status === "ready" || status === "uploaded") {
        return
      }
    } catch (error) {
      lastError = error
    }

    await delay(OPENAI_FILE_POLL_INTERVAL_MS)
  }

  if (lastError) {
    const status = Number((lastError as any)?.status) || 504
    throw Object.assign(new Error("OpenAI file processing timeout"), {
      status,
      details: {
        endpoint: "files.status",
        status: lastStatus || "timeout",
        body:
          (lastError as any)?.details ||
          (lastError instanceof Error ? lastError.message : String(lastError)),
      },
    })
  }

  throw Object.assign(new Error("OpenAI file processing timeout"), {
    status: 504,
    details: { endpoint: "files.status", status: lastStatus || "timeout" },
  })
}

async function fetchFileMetadata(fileId: string) {
  const response = await fetch(`${API_BASE}/files/${fileId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
    },
  })

  if (!response.ok) {
    const payload = await readErrorPayload(response)
    throw Object.assign(new Error("OpenAI file status request failed"), {
      status: response.status,
      details: { endpoint: "files.status", status: response.status, body: payload },
    })
  }

  return response.json()
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function callOpenAiResponses(content: OpenAiContent) {
  const response = await fetch(`${API_BASE}/responses`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      model: ACTIVE_OPENAI_MODEL,
      input: [
        {
          role: "user",
          content,
        },
      ],
      response_format: { type: "json_schema", json_schema: SCHEMA },
      temperature: TEMPERATURE,
      seed: OPENAI_SEED,
    }),
  })

  if (!response.ok) {
    const payload = await readErrorPayload(response)
    throw Object.assign(new Error("OpenAI responses request failed"), {
      status: response.status,
      details: { endpoint: "responses", status: response.status, body: payload },
    })
  }

  return response.json()
}

async function callOpenAiChatCompletions(content: OpenAiContent) {
  const response = await fetch(`${API_BASE}/chat/completions`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      model: ACTIVE_OPENAI_MODEL,
      messages: [
        {
          role: "user",
          content,
        },
      ],
      response_format: { type: "json_schema", json_schema: SCHEMA },
      temperature: TEMPERATURE,
      seed: OPENAI_SEED,
    }),
  })

  if (!response.ok) {
    const payload = await readErrorPayload(response)
    throw Object.assign(new Error("OpenAI chat request failed"), {
      status: response.status,
      details: { endpoint: "chat.completions", status: response.status, body: payload },
    })
  }

  return response.json()
}

function extractTextPayload(payload: any): string {
  if (!payload) return "{}"

  if (typeof payload.output_text === "string") {
    return payload.output_text
  }

  if (typeof payload.response_text === "string") {
    return payload.response_text
  }

  const firstOutput = payload.output?.[0]?.content?.[0]
  if (firstOutput?.type === "output_text" && typeof firstOutput.text === "string") {
    return firstOutput.text
  }

  const messageContent = payload.choices?.[0]?.message?.content
  if (typeof messageContent === "string") {
    return messageContent
  }

  if (Array.isArray(messageContent)) {
    const textEntry = messageContent.find(
      (part: any) => part?.type === "text" && typeof part.text === "string"
    )
    if (textEntry) {
      return textEntry.text as string
    }
  }

  return "{}"
}

function sanitiseArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0)
}

function sanitiseSubscores(value: unknown): Record<string, number> | undefined {
  if (!value || typeof value !== "object") return undefined

  const result: Record<string, number> = {}
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      result[key] = clampScore(raw)
    }
  }

  return Object.keys(result).length ? result : undefined
}

function clampScore(value: unknown): number {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return 0
  return Math.min(100, Math.max(0, numeric))
}

function shouldFallbackToChat(error: unknown): boolean {
  if (!error || typeof error !== "object") return false
  const status = "status" in error ? Number((error as any).status) : undefined
  if (status === 404 || status === 400 || status === 501) {
    return true
  }
  const details = (error as any).details
  if (details) {
    const detailString = typeof details === "string" ? details : JSON.stringify(details)
    if (/responses?/i.test(detailString) && /not\s+(found|available|enabled)/i.test(detailString)) {
      return true
    }
  }
  const message = (error as any).message ? String((error as any).message) : ""
  return /responses?/i.test(message)
}

function serialiseDetails(details: unknown): string {
  if (!details) return ""
  if (typeof details === "string") return details
  try {
    return JSON.stringify(details)
  } catch {
    return String(details)
  }
}

export async function scoreBufferWithOpenAI(opts: {
  buffer: Buffer
  mimetype: string
  filename: string
}): Promise<AiResult> {
  const isPdf = /pdf$/i.test(opts.mimetype) || /\.pdf$/i.test(opts.filename)
  const fileMeta = {
    name: opts.filename,
    mimetype: opts.mimetype,
    size: opts.buffer?.length ?? 0,
  }
  const branch = isPdf ? "pdf" : "image"

  if (!API_KEY) {
    console.warn(
      JSON.stringify({
        event: "physical.score",
        level: "warn",
        provider: "openai",
        reason: "missing-openai-api-key",
        branch,
        file: fileMeta,
      })
    )
    throw Object.assign(new Error("OPENAI_API_KEY not configured"), {
      status: 400,
      details: {
        status: 400,
        message: "OPENAI_API_KEY not configured",
        hint: "Set OPENAI_API_KEY or disable USE_OPENAI_SCORER",
        model: ACTIVE_OPENAI_MODEL,
        branch,
        file: fileMeta,
      },
    })
  }

  const signals = !isPdf ? await computeQualitySignals(opts.buffer, opts.mimetype) : {}

  try {
    const pdfFileId = isPdf ? await uploadPdf(opts.buffer, opts.filename) : undefined
    const imageDataUrl = !isPdf
      ? bufferToDataUrl(opts.buffer, opts.mimetype || "image/jpeg")
      : undefined

    const documentContext = buildDocumentContext({
      buffer: opts.buffer,
      mimetype: opts.mimetype,
      filename: opts.filename,
      branch,
    })

    const responsesContent: OpenAiContent = isPdf
      ? [
          { type: "input_text", text: buildPrompt(signals, documentContext) },
          { type: "input_file", file_id: pdfFileId! },
        ]
      : [
          { type: "input_text", text: buildPrompt(signals, documentContext) },
          { type: "input_image", image_url: { url: imageDataUrl! } },
        ]

    const chatContent: OpenAiContent = isPdf
      ? [
          { type: "text", text: buildPrompt(signals, documentContext) },
          { type: "input_file", file_id: pdfFileId! },
        ]
      : [
          { type: "text", text: buildPrompt(signals, documentContext) },
          { type: "image_url", image_url: { url: imageDataUrl! } },
        ]

    let payload: any
    try {
      payload = await callOpenAiResponses(responsesContent)
    } catch (error) {
      if (!shouldFallbackToChat(error)) {
        throw error
      }
      payload = await callOpenAiChatCompletions(chatContent)
    }

    const text = extractTextPayload(payload)

    let parsed: Partial<AiResult> = {}
    try {
      parsed = JSON.parse(text)
    } catch {
      console.warn(
        JSON.stringify({
          event: "physical.score",
          level: "warn",
          provider: "openai",
          reason: "invalid-json",
          sample: text.slice(0, 240),
        })
      )
      parsed = {}
    }

    const qualitySignals = signals && Object.keys(signals).length ? signals : undefined

    const result: AiResult = {
      score: clampScore(parsed.score),
      reasons: sanitiseArray(parsed.reasons),
      flags: sanitiseArray(parsed.flags),
      suggestions: sanitiseArray(parsed.suggestions),
      confidence: clampConfidence(parsed.confidence),
      provider: "openai",
      model: ACTIVE_OPENAI_MODEL,
      qualitySignals,
    }

    if (!Number.isFinite(result.score)) {
      result.score = 0
    }

    if (!result.reasons.length) {
      result.reasons = []
    }

    if (!result.flags.length) {
      result.flags = []
    }

    if (!result.suggestions.length) {
      result.suggestions = []
    }

    const subscores = sanitiseSubscores(parsed.subscores)
    if (subscores) {
      result.subscores = subscores
    }

    if (qualitySignals) {
      const baseline = result.subscores ?? {}
      if (qualitySignals.quality_blur !== undefined && baseline.quality_blur === undefined) {
        baseline.quality_blur = clampScore(qualitySignals.quality_blur)
      }
      if (qualitySignals.quality_glare !== undefined && baseline.quality_glare === undefined) {
        baseline.quality_glare = clampScore(qualitySignals.quality_glare)
      }
      if (qualitySignals.quality_skew !== undefined && baseline.quality_skew === undefined) {
        baseline.quality_skew = clampScore(qualitySignals.quality_skew)
      }
      if (Object.keys(baseline).length > 0) {
        result.subscores = baseline
      }
    }

    return result
  } catch (error: any) {
    const status = Number.isFinite(Number(error?.status))
      ? Number(error.status)
      : Number(error?.response?.status) || 502
    const upstream = error?.details || error?.response?.data || error?.message || error

    throw Object.assign(new Error(error?.message || "OpenAI error"), {
      status,
      details: {
        status,
        message: error?.message || "Unexpected OpenAI error",
        model: ACTIVE_OPENAI_MODEL,
        branch,
        file: fileMeta,
        upstream: serialiseDetails(upstream),
      },
    })
  }
}

function buildPrompt(signals: QualitySignals | undefined, context?: DocumentContext): string {
  let prompt = buildPromptCore(signals)
  if (context) {
    prompt += `\nDocument metadata: ${context.metadataSummary}.`
    if (context.textSnippet) {
      prompt += `\nExtracted text snippet from the uploaded file (may contain noise): ${context.textSnippet}`
    }
    if (context.imageFingerprint) {
      prompt += `\nImage fingerprint preview (first characters of base64, do not treat as the full image): ${context.imageFingerprint}`
    }
    prompt += `\nUse the attached document bytes for your inspection before producing the JSON verdict.`
  }
  return prompt
}

function buildPromptCore(signals: QualitySignals | undefined): string {
  const base = PROMPT
  if (!signals || Object.keys(signals).length === 0) {
    return base
  }

  const summary: Record<string, unknown> = {
    ...(signals.quality_blur !== undefined && {
      quality_blur: signals.quality_blur,
    }),
    ...(signals.quality_glare !== undefined && {
      quality_glare: signals.quality_glare,
    }),
    ...(signals.quality_skew !== undefined && {
      quality_skew: signals.quality_skew,
    }),
  }

  if (signals.metrics) {
    summary.metrics = signals.metrics
  }

  return `${base}\nLocal quality metrics (0-100 higher=better): ${JSON.stringify(summary)}`
}

function clampConfidence(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return undefined
  return Math.min(1, Math.max(0, numeric))
}

async function computeQualitySignals(buffer: Buffer, mimetype: string): Promise<QualitySignals> {
  try {
    const sizeBytes = buffer?.length ?? 0
    const sizeKb = sizeBytes / 1024
    const previewLimit = 2 * 1024 * 1024
    const previewDataUrl =
      sizeBytes > 0 && sizeBytes <= previewLimit
        ? bufferToDataUrl(buffer, mimetype || "image/jpeg")
        : undefined

    const blurScore = sizeKb <= 50 ? 30 : sizeKb >= 400 ? 90 : 30 + ((sizeKb - 50) / 350) * 60
    const glareScore = sizeKb <= 80 ? 55 : sizeKb >= 600 ? 85 : 55 + ((sizeKb - 80) / 520) * 30

    return {
      previewDataUrl,
      quality_blur: clampScore(Math.round(blurScore)),
      quality_glare: clampScore(Math.round(glareScore)),
    }
  } catch (error) {
    console.warn("[openai] Failed to compute quality signals", error)
    return {}
  }
}

function buildDocumentContext(options: {
  buffer: Buffer
  mimetype: string
  filename: string
  branch: string
}): DocumentContext {
  const { buffer, mimetype, filename, branch } = options
  const hash = createHash("sha256").update(buffer).digest("hex")
  const metaParts = [
    `filename=${filename}`,
    `mimetype=${mimetype}`,
    `bytes=${buffer.length}`,
    `sha256=${hash.slice(0, 32)}`,
  ]

  const context: DocumentContext = {
    metadataSummary: metaParts.join(", "),
  }

  if (branch === "pdf") {
    const snippet = extractPdfTextSnippet(buffer)
    if (snippet) {
      context.textSnippet = snippet
    }
  } else {
    const fingerprint = buildImageFingerprint(buffer)
    if (fingerprint) {
      context.imageFingerprint = fingerprint
    }
  }

  return context
}

function extractPdfTextSnippet(buffer: Buffer, maxLength = 800): string | undefined {
  if (!buffer?.length) return undefined
  const raw = buffer.toString("latin1")
  const matches = raw.match(/[\t\n\r\x20-\x7e]{4,}/g)
  if (!matches) {
    return undefined
  }
  const combined = matches.join(" ").replace(/\s+/g, " ").trim()
  if (!combined) {
    return undefined
  }
  return combined.slice(0, maxLength)
}

function buildImageFingerprint(buffer: Buffer, maxLength = 200): string | undefined {
  if (!buffer?.length) return undefined
  const base64 = buffer.toString("base64")
  if (!base64) return undefined
  return base64.slice(0, maxLength)
}
