import "./env"

export type AiResult = {
  score: number
  reasons: string[]
  flags: string[]
  suggestions: string[]
  subscores?: Record<string, number>
  confidence?: number
  provider: "cnn" | "fallback" | "heuristic"
  model: string
  qualitySignals?: { previewDataUrl?: string }
}

type MonteCarloResponse = {
  prediction: number
  prediction_label: string
  confidence: number
  confidence_interval: {
    mean: number
    lower_bound: number
    upper_bound: number
    confidence_level: number
  }
  monte_carlo_stats: {
    num_samples: number
    agreement_rate: number
    std_dev: number
    class_probabilities: { fake: number; real: number }
  }
  timestamp: string
  result_id: string
  document_id: string
  gcs_path: string
}

type PredictResponse = {
  prediction: number
  prediction_label: string
  confidence: number
  confidence_interval: {
    mean: number
    lower_bound: number
    upper_bound: number
    confidence_level: number
  }
  timestamp: string
  result_id: string
  document_id: string
  gcs_path: string
}

const FASTAPI_BASE = (process.env.FASTAPI_URL || "http://localhost:8000").replace(/\/+$/, "")
const CNN_MODEL_NAME = "best_real_fake_resnet18"
const USE_MONTE_CARLO = String(process.env.USE_MONTE_CARLO || "true").toLowerCase() === "true"
const MC_SAMPLES = Number(process.env.MC_SAMPLES) || 30

export const ACTIVE_CNN_MODEL = CNN_MODEL_NAME

export async function scoreWithCNN(opts: {
  buffer: Buffer
  mimetype: string
  filename: string
}): Promise<AiResult> {
  const endpoint = USE_MONTE_CARLO ? "/monte_carlo" : "/predict"
  const url = new URL(endpoint, FASTAPI_BASE)

  if (USE_MONTE_CARLO) {
    url.searchParams.set("num_samples", String(MC_SAMPLES))
  }

  const formData = new FormData()
  const blob = new Blob([opts.buffer], { type: opts.mimetype || "application/octet-stream" })
  formData.append("file", blob, opts.filename || "upload")

  let response: Response
  try {
    response = await fetch(url.toString(), { method: "POST", body: formData })
  } catch (networkError: any) {
    console.error("[cnn] FastAPI unreachable:", networkError?.message)
    throw Object.assign(new Error("CNN backend unreachable"), {
      status: 502,
      details: { hint: `Ensure FastAPI is running at ${FASTAPI_BASE}`, endpoint },
    })
  }

  if (!response.ok) {
    let detail = "Unknown error"
    try {
      const body = await response.json()
      detail = body?.detail || JSON.stringify(body)
    } catch {
      detail = await response.text().catch(() => `HTTP ${response.status}`)
    }
    console.error(`[cnn] FastAPI ${endpoint} returned ${response.status}: ${detail}`)
    throw Object.assign(new Error(`CNN scoring failed: ${detail}`), {
      status: response.status,
    })
  }

  const data = await response.json()
  return USE_MONTE_CARLO
    ? mapMonteCarloToAiResult(data as MonteCarloResponse)
    : mapPredictToAiResult(data as PredictResponse)
}

function mapMonteCarloToAiResult(mc: MonteCarloResponse): AiResult {
  const score = Math.round(mc.confidence * 100)
  const ci = mc.confidence_interval
  const stats = mc.monte_carlo_stats
  const isReal = mc.prediction === 1

  const reasons: string[] = []
  reasons.push(
    isReal
      ? `CNN classified this document as authentic (${mc.prediction_label}) with ${score}% confidence.`
      : `CNN flagged this document as potentially inauthentic (${mc.prediction_label}) with ${score}% confidence.`
  )

  if (stats.agreement_rate >= 0.9) {
    reasons.push(`High consistency: ${Math.round(stats.agreement_rate * 100)}% of ${stats.num_samples} augmented samples agreed.`)
  } else if (stats.agreement_rate >= 0.7) {
    reasons.push(`Moderate consistency: ${Math.round(stats.agreement_rate * 100)}% agreement across ${stats.num_samples} samples.`)
  } else {
    reasons.push(`Low consistency: only ${Math.round(stats.agreement_rate * 100)}% agreement — result may be unreliable.`)
  }

  reasons.push(`95% CI: ${Math.round(ci.lower_bound * 100)}%–${Math.round(ci.upper_bound * 100)}%.`)

  const flags: string[] = []
  if (!isReal) flags.push("classified-fake")
  if (stats.agreement_rate < 0.7) flags.push("low-agreement")
  if (mc.confidence < 0.6) flags.push("low-confidence")
  if (stats.std_dev > 0.15) flags.push("high-variance")

  const suggestions: string[] = []
  if (mc.confidence < 0.7) suggestions.push("Submit a higher-resolution scan for more reliable classification.")
  if (stats.agreement_rate < 0.8) suggestions.push("Augmentation results are inconsistent — manual review recommended.")
  if (!isReal) suggestions.push("Document flagged as potentially fake. Verify against original sources.")

  return {
    score,
    reasons,
    flags,
    suggestions,
    subscores: {
      authenticity: score,
      agreement_rate: Math.round(stats.agreement_rate * 100),
      stability: Math.round(Math.max(0, (1 - stats.std_dev) * 100)),
      prob_real: Math.round(stats.class_probabilities.real * 100),
      prob_fake: Math.round(stats.class_probabilities.fake * 100),
    },
    confidence: mc.confidence,
    provider: "cnn",
    model: CNN_MODEL_NAME,
  }
}

function mapPredictToAiResult(pred: PredictResponse): AiResult {
  const score = Math.round(pred.confidence * 100)
  const isReal = pred.prediction === 1

  return {
    score,
    reasons: [
      isReal
        ? `CNN classified this document as authentic (${pred.prediction_label}) with ${score}% confidence.`
        : `CNN flagged this document as potentially inauthentic (${pred.prediction_label}) with ${score}% confidence.`,
      `95% CI: ${Math.round(pred.confidence_interval.lower_bound * 100)}%–${Math.round(pred.confidence_interval.upper_bound * 100)}%.`,
    ],
    flags: [...(!isReal ? ["classified-fake"] : []), ...(pred.confidence < 0.6 ? ["low-confidence"] : [])],
    suggestions: pred.confidence < 0.7 ? ["Submit a higher-resolution scan for more reliable results."] : [],
    confidence: pred.confidence,
    provider: "cnn",
    model: CNN_MODEL_NAME,
  }
}
