import "./env"

export type CnnResult = {
  confidence: number                    // 0-1 from CNN softmax
  confidenceInterval: {
    lower: number                       // 0-1
    upper: number                       // 0-1
  }
  monteCarloStats?: {
    numSamples: number
    agreementRate: number               // 0-1
    stdDev: number
  }
  provider: "cnn" | "heuristic"
  model: string
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
const CNN_MODEL_NAME = "resnet18-real-fake"
const USE_MONTE_CARLO = String(process.env.USE_MONTE_CARLO || "true").toLowerCase() === "true"
const MC_SAMPLES = Number(process.env.MC_SAMPLES) || 30

export const ACTIVE_CNN_MODEL = CNN_MODEL_NAME

export async function scoreWithCNN(opts: {
  buffer: Buffer
  mimetype: string
  filename: string
}): Promise<CnnResult> {
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
    ? mapMonteCarlo(data as MonteCarloResponse)
    : mapPredict(data as PredictResponse)
}

function mapMonteCarlo(mc: MonteCarloResponse): CnnResult {
  return {
    confidence: mc.confidence,
    confidenceInterval: {
      lower: mc.confidence_interval.lower_bound,
      upper: mc.confidence_interval.upper_bound,
    },
    monteCarloStats: {
      numSamples: mc.monte_carlo_stats.num_samples,
      agreementRate: mc.monte_carlo_stats.agreement_rate,
      stdDev: mc.monte_carlo_stats.std_dev,
    },
    provider: "cnn",
    model: CNN_MODEL_NAME,
  }
}

function mapPredict(pred: PredictResponse): CnnResult {
  return {
    confidence: pred.confidence,
    confidenceInterval: {
      lower: pred.confidence_interval.lower_bound,
      upper: pred.confidence_interval.upper_bound,
    },
    provider: "cnn",
    model: CNN_MODEL_NAME,
  }
}
