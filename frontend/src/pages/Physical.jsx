import { useEffect, useMemo, useRef, useState } from "react"
import PropTypes from "prop-types"
import { Link } from "react-router-dom"
import {
  ChevronLeft,
  ChevronRight,
  FilePlus2,
  Files,
  Printer,
  Save,
  ScanLine,
  Trash2,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { normalizeError } from "@/utils/normalizeError"
import { API_BASE } from "@/api/verify"

const createEmptyResult = () => ({
  confidence: null,
  confidenceInterval: null,
  monteCarloStats: null,
  analysisId: "",
  requestId: "",
  model: "",
  provider: "",
  elapsedMs: null,
  error: null,
})

const SUPPORTED_IMAGE_EXT = /\.(png|jpe?g|jpg|webp|gif|heic)$/i

const isImageFile = (file) => {
  if (!file) return false
  const type = file.type?.toLowerCase() ?? ""
  if (type.startsWith("image/")) return true
  return SUPPORTED_IMAGE_EXT.test(file.name || "")
}
const isPdfFile = (file) => {
  if (!file) return false
  const type = file.type?.toLowerCase() ?? ""
  if (type.includes("pdf")) return true
  return /\.pdf$/i.test(file.name || "")
}

const formatFileSize = (size) => {
  if (typeof size !== "number" || !Number.isFinite(size) || size < 0) return ""
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`
  }
  if (size >= 1024) {
    return `${(size / 1024).toFixed(0)} KB`
  }
  return `${size} B`
}

// formatScore removed — raw confidence values displayed directly

// clampScore removed — no score interpretation

const isNetworkFailure = (error) =>
  error instanceof TypeError && /failed to fetch/i.test(error.message || "")

const buildNetworkErrorInfo = (message, title) =>
  normalizeError(message, {
    fallbackTitle: title,
    fallbackMessage: message,
    debug: JSON.stringify({ apiBase: API_BASE }, null, 2),
  })

// Section component removed — no reasons/flags/suggestions to display

const parseScoreResponse = (payload) => ({
  confidence:
    typeof payload.confidence === "number"
      ? Math.min(1, Math.max(0, payload.confidence))
      : null,
  confidenceInterval:
    payload.confidenceInterval && typeof payload.confidenceInterval === "object"
      ? {
          lower: typeof payload.confidenceInterval.lower === "number" ? payload.confidenceInterval.lower : null,
          upper: typeof payload.confidenceInterval.upper === "number" ? payload.confidenceInterval.upper : null,
        }
      : null,
  monteCarloStats:
    payload.monteCarloStats && typeof payload.monteCarloStats === "object"
      ? payload.monteCarloStats
      : null,
  analysisId: typeof payload.analysisId === "string" ? payload.analysisId : "",
  requestId: typeof payload.requestId === "string" ? payload.requestId : "",
  model: typeof payload.model === "string" ? payload.model : "",
  provider: typeof payload.provider === "string" ? payload.provider : "",
  elapsedMs:
    typeof payload.elapsedMs === "number" && Number.isFinite(payload.elapsedMs)
      ? payload.elapsedMs
      : null,
  error: typeof payload.error === "string" ? payload.error : null,
})

export default function Physical({ initialTab = "verification", comparisonOnly = false, embedded = false }) {
  const [activeTab, setActiveTab] = useState(initialTab)

  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isGeneratingReport, setIsGeneratingReport] = useState(false)
  const [errorInfo, setErrorInfo] = useState(null)
  const [result, setResult] = useState(createEmptyResult)
  const [resultTimestamp, setResultTimestamp] = useState("")
  const [reportUrl, setReportUrl] = useState("")
  const [showWhy, setShowWhy] = useState(false)
  const [copyState, setCopyState] = useState("idle")

  const [knownFile, setKnownFile] = useState(null)
  const [questionedFile, setQuestionedFile] = useState(null)
  const [comparisonNotes, setComparisonNotes] = useState("")
  const [caseName, setCaseName] = useState("")
  const [comparisonResult, setComparisonResult] = useState(null)
  const [comparisonError, setComparisonError] = useState(null)
  const [isComparing, setIsComparing] = useState(false)
  const [knownPreview, setKnownPreview] = useState(null)
  const [questionedPreview, setQuestionedPreview] = useState(null)
  const [knownZoom, setKnownZoom] = useState(116)
  const [questionedZoom, setQuestionedZoom] = useState(149)

  const [health, setHealth] = useState({ status: "loading", data: null })

  const fileInputRef = useRef(null)

  useEffect(() => {
    if (comparisonOnly && activeTab !== "comparison") {
      setActiveTab("comparison")
    }
  }, [comparisonOnly, activeTab])

  const confidencePercent = useMemo(() => {
    if (typeof result.confidence !== "number" || Number.isNaN(result.confidence)) {
      return null
    }
    return `${(result.confidence * 100).toFixed(1)}%`
  }, [result.confidence])

  const ciBounds = useMemo(() => {
    if (!result.confidenceInterval) return null
    const { lower, upper } = result.confidenceInterval
    if (typeof lower !== "number" || typeof upper !== "number") return null
    return `${(lower * 100).toFixed(1)}% – ${(upper * 100).toFixed(1)}%`
  }, [result.confidenceInterval])

  useEffect(() => {
    if (!file || !isImageFile(file)) {
      setPreviewUrl(null)
      return undefined
    }

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  useEffect(() => {
    return () => {
      if (reportUrl) {
        URL.revokeObjectURL(reportUrl)
      }
    }
  }, [reportUrl])

  useEffect(() => {
    if (!knownFile || !isImageFile(knownFile)) {
      setKnownPreview(null)
      return undefined
    }

    const url = URL.createObjectURL(knownFile)
    setKnownPreview(url)

    return () => URL.revokeObjectURL(url)
  }, [knownFile])

  useEffect(() => {
    if (!questionedFile || !isImageFile(questionedFile)) {
      setQuestionedPreview(null)
      return undefined
    }

    const url = URL.createObjectURL(questionedFile)
    setQuestionedPreview(url)

    return () => URL.revokeObjectURL(url)
  }, [questionedFile])

  // subscores and qualitySignals removed — raw CNN output only

  useEffect(() => {
    setCopyState("idle")
  }, [errorInfo?.details])

  const previewSource = useMemo(() => {
    if (previewUrl) return previewUrl
    return null
  }, [previewUrl])

  useEffect(() => {
    let isMounted = true

    const fetchHealth = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/physical/health`)
        const contentType = response.headers.get("content-type")?.toLowerCase() ?? ""
        const rawText = await response.text()

        if (!isMounted) return

        if (!response.ok) {
          const payload = contentType.includes("application/json")
            ? (() => {
                try {
                  return rawText ? JSON.parse(rawText) : {}
                } catch {
                  return null
                }
              })()
            : null

          setHealth({
            status: "error",
            error: normalizeError(payload || rawText || "Health check failed", {
              fallbackTitle: "Health check failed",
              fallbackMessage: "Unable to reach the scorer health endpoint.",
              status: response.status,
              debug: payload ? JSON.stringify(payload, null, 2) : rawText,
            }),
          })
          return
        }

        if (!contentType.includes("application/json")) {
          setHealth({
            status: "error",
            error: normalizeError(rawText || "Unexpected response", {
              fallbackTitle: "Health check failed",
              fallbackMessage: "Health endpoint returned unexpected data.",
              status: response.status,
              debug: rawText,
            }),
          })
          return
        }

        const payload = rawText ? JSON.parse(rawText) : {}
        setHealth({ status: "success", data: payload })
      } catch (fetchError) {
        if (!isMounted) return
        setHealth({
          status: "error",
          error: normalizeError(fetchError, {
            fallbackTitle: "Health check failed",
            fallbackMessage: "Unable to reach the scorer health endpoint.",
          }),
        })
      }
    }

    fetchHealth()

    return () => {
      isMounted = false
    }
  }, [])

  const scoreFile = async (selectedFile) => {
    const formData = new FormData()
    formData.append("file", selectedFile, selectedFile.name)

    const response = await fetch(`${API_BASE}/api/physical/score`, {
      method: "POST",
      body: formData,
    })

    const contentType = response.headers.get("content-type")?.toLowerCase() ?? ""
    const rawText = await response.text().catch(() => "")
    const payload = contentType.includes("application/json") && rawText
      ? JSON.parse(rawText)
      : null

    if (!response.ok) {
      throw normalizeError(payload || rawText || "Scoring failed", {
        fallbackTitle: "Scoring failed",
        fallbackMessage: "The scorer could not process this document.",
        status: response.status,
        debug: payload ? JSON.stringify(payload, null, 2) : rawText,
      })
    }

    if (!payload || typeof payload !== "object") {
      throw normalizeError(rawText || "Unexpected response", {
        fallbackTitle: "Scoring failed",
        fallbackMessage: "Unexpected response from server.",
        status: response.status,
        debug: rawText,
      })
    }

    return parseScoreResponse(payload)
  }

  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0] ?? null
    setFile(nextFile)
    setResult(createEmptyResult())
    setResultTimestamp("")
    setErrorInfo(null)
    setShowWhy(false)
    if (reportUrl) {
      URL.revokeObjectURL(reportUrl)
      setReportUrl("")
    }
  }

  const handleReset = () => {
    setFile(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    setResult(createEmptyResult())
    setResultTimestamp("")
    setErrorInfo(null)
    setShowWhy(false)
    if (reportUrl) {
      URL.revokeObjectURL(reportUrl)
      setReportUrl("")
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!file) {
      setErrorInfo(
        normalizeError("Please choose an image or PDF of the document first.", {
          fallbackTitle: "Scoring failed",
          fallbackMessage: "Please choose an image or PDF of the document first.",
        })
      )
      return
    }

    setIsSubmitting(true)
    setErrorInfo(null)
    setResult(createEmptyResult())
    setResultTimestamp("")

    try {
      const scored = await scoreFile(file)
      setResult(scored)
      setResultTimestamp(new Date().toISOString())
      setShowWhy(true)
    } catch (submissionError) {
      if (isNetworkFailure(submissionError)) {
        setErrorInfo(
          buildNetworkErrorInfo(
            "Unable to reach the scoring service. Please make sure the API is running and accessible.",
            "Scoring failed"
          )
        )
        return
      }
      setErrorInfo(
        normalizeError(submissionError, {
          fallbackTitle: "Scoring failed",
          fallbackMessage: "An unexpected error occurred while scoring the document.",
        })
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleRunComparison = async () => {
    if (!knownFile || !questionedFile) {
      setComparisonError(
        normalizeError("Upload both the original and questioned documents before comparing.", {
          fallbackTitle: "Comparison failed",
          fallbackMessage: "Upload both files before starting comparison.",
        })
      )
      return
    }

    setIsComparing(true)
    setComparisonError(null)
    setComparisonResult(null)

    try {
      const [knownResult, questionedResult] = await Promise.all([
        scoreFile(knownFile),
        scoreFile(questionedFile),
      ])

      const knownConf = typeof knownResult.confidence === "number" ? knownResult.confidence : null
      const questionedConf = typeof questionedResult.confidence === "number" ? questionedResult.confidence : null

      const formatConf = (v) => v !== null ? `${(v * 100).toFixed(1)}%` : "N/A"
      const formatCI = (r) => {
        if (!r.confidenceInterval) return "N/A"
        return `${(r.confidenceInterval.lower * 100).toFixed(1)}% – ${(r.confidenceInterval.upper * 100).toFixed(1)}%`
      }

      const reportLines = [
        `Case: ${caseName || "Unlabeled case"}`,
        `Generated: ${new Date().toLocaleString()}`,
        `Original file: ${knownFile.name}`,
        `Questioned file: ${questionedFile.name}`,
        "",
        `Original — Confidence: ${formatConf(knownConf)}, CI: ${formatCI(knownResult)}`,
        `Questioned — Confidence: ${formatConf(questionedConf)}, CI: ${formatCI(questionedResult)}`,
      ]

      if (comparisonNotes.trim()) {
        reportLines.push("", `Investigator notes: ${comparisonNotes.trim()}`)
      }

      setComparisonResult({
        summary: "Comparison complete. Review confidence values for each document.",
        reportText: reportLines.join("\n"),
        knownResult,
        questionedResult,
      })
    } catch (compareError) {
      if (isNetworkFailure(compareError)) {
        setComparisonError(
          buildNetworkErrorInfo(
            "Unable to reach the scoring service for comparison. Please ensure the API is running.",
            "Comparison failed"
          )
        )
      } else {
        setComparisonError(
          normalizeError(compareError, {
            fallbackTitle: "Comparison failed",
            fallbackMessage: "An unexpected error occurred while comparing documents.",
          })
        )
      }
    } finally {
      setIsComparing(false)
    }
  }

  const handleDownloadComparisonReport = () => {
    if (!comparisonResult?.reportText) return
    const blob = new Blob([comparisonResult.reportText], { type: "text/plain;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `kwiddex-comparison-${caseName || "report"}.txt`
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }

  const handleResetComparison = () => {
    setKnownFile(null)
    setQuestionedFile(null)
    setComparisonNotes("")
    setCaseName("")
    setComparisonResult(null)
    setComparisonError(null)
    setKnownZoom(116)
    setQuestionedZoom(149)
  }

  const handleGenerateReport = async () => {
    if (!result.analysisId) {
      setErrorInfo(
        normalizeError("Run a successful score before generating a report.", {
          fallbackTitle: "Report unavailable",
          fallbackMessage: "Run a successful score before generating a report.",
        })
      )
      return
    }

    setIsGeneratingReport(true)
    setErrorInfo(null)

    try {
      const response = await fetch(`${API_BASE}/api/physical/report`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ analysisId: result.analysisId }),
      })

      const contentType = response.headers.get("content-type")?.toLowerCase() ?? ""
      if (!response.ok) {
        const raw = await response.text()
        setErrorInfo(
          normalizeError(raw || "Unable to generate report", {
            fallbackTitle: "Report generation failed",
            fallbackMessage: "Unable to generate report.",
            status: response.status,
            debug: raw,
          })
        )
        return
      }

      if (!contentType.includes("application/pdf")) {
        const unexpected = await response.text()
        setErrorInfo(
          normalizeError(unexpected || "Unexpected response", {
            fallbackTitle: "Report generation failed",
            fallbackMessage: "Unexpected response while generating report.",
            debug: unexpected,
          })
        )
        return
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setReportUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev)
        return url
      })

      const anchor = document.createElement("a")
      anchor.href = url
      anchor.download = `kwiddex-report-${result.analysisId}.pdf`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
    } catch (reportError) {
      if (isNetworkFailure(reportError)) {
        setErrorInfo(
          buildNetworkErrorInfo(
            "Unable to reach the report service. Please make sure the API is running and accessible.",
            "Report generation failed"
          )
        )
        return
      }
      setErrorInfo(
        normalizeError(reportError, {
          fallbackTitle: "Report generation failed",
          fallbackMessage: "Failed to generate the explanation report.",
        })
      )
    } finally {
      setIsGeneratingReport(false)
    }
  }

  const handleCopyDetails = async () => {
    if (!errorInfo?.details || typeof navigator === "undefined" || !navigator.clipboard) {
      setCopyState("error")
      setTimeout(() => setCopyState("idle"), 2000)
      return
    }

    try {
      await navigator.clipboard.writeText(errorInfo.details)
      setCopyState("copied")
      setTimeout(() => setCopyState("idle"), 2000)
    } catch {
      setCopyState("error")
      setTimeout(() => setCopyState("idle"), 2000)
    }
  }

  const healthVariant = useMemo(() => {
    if (health.status === "success" && health.data?.ok) {
      if (health.data.provider === "cnn") return "bg-emerald-500"
      return "bg-amber-400"
    }
    if (health.status === "error") return "bg-destructive"
    return "bg-muted-foreground/60"
  }, [health])

  const healthLabel = useMemo(() => {
    if (health.status === "loading") return "Checking scorer status…"
    if (health.status === "success" && health.data?.ok) {
      if (health.data.provider === "cnn") {
        return `CNN online (model ${health.data.model || "unknown"})`
      }
      return "Scorer available (non-CNN)"
    }
    if (health.status === "error") {
      return health.error?.message || "Scorer unreachable"
    }
    return "Scorer status unknown"
  }, [health])

  return (
    <div className={`${embedded ? "text-base-color" : "min-h-screen bg-base text-base-color flex flex-col"}`}>
      {!embedded && (
      <header className="border-b border-border bg-base/80 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="text-lg font-semibold text-blue-600 hover:text-blue-500">
            &larr; Back to Home
          </Link>
          <span className="text-sm text-muted-foreground">KwiddeX Intelligence</span>
        </div>
      </header>
      )}

      <main className={`${embedded ? "" : "flex-1"} px-4 py-10`}>
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
            <div className="flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${healthVariant}`} title={healthLabel} />
              <h1 className="text-2xl font-semibold tracking-tight text-base-color">"Physical document checks"</h1>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              "Upload a physical document image or PDF to get a CNN confidence analysis."
            </p>

          </div>

          {true ? (
            <div className="grid gap-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <form className="space-y-6 rounded-2xl border border-border bg-card p-6 shadow-sm" onSubmit={handleSubmit}>
                <p className="text-sm text-muted-foreground">
                  Scan or upload a physical document image/PDF to get a CNN confidence analysis.
                </p>

                <label
                  htmlFor="physical-file"
                  className="block rounded-xl border border-dashed border-border bg-muted/40 p-6 text-center transition hover:border-blue-500 hover:bg-muted"
                >
                  <input
                    id="physical-file"
                    name="file"
                    type="file"
                    accept="image/*,application/pdf,.pdf"
                    className="sr-only"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                  />
                  <div className="space-y-2">
                    <p className="text-base font-medium text-base-color">
                      {file ? file.name : "Drop an image or PDF, or click to browse"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supported formats: JPEG, PNG, HEIC, or PDF up to 15&nbsp;MB.
                    </p>
                  </div>
                </label>

                {previewSource ? (
                  <div className="overflow-hidden rounded-xl border border-border">
                    <img src={previewSource} alt="Document preview" className="h-64 w-full object-cover" />
                  </div>
                ) : (
                  file && (
                    <div className="rounded-xl border border-border bg-muted/40 p-4 text-left text-sm">
                      <p className="font-medium text-base-color">{file.name}</p>
                      <p className="text-muted-foreground">
                        {isPdfFile(file) ? "PDF document" : file.type || "File"}
                        {file.size ? ` • ${formatFileSize(file.size)}` : ""}
                      </p>
                    </div>
                  )
                )}

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="submit"
                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-blue-300"
                    disabled={isSubmitting || !file}
                  >
                    {isSubmitting ? "Scoring…" : "Check document"}
                  </button>
                  {file && (
                    <button
                      type="button"
                      className="text-sm font-medium text-muted-foreground transition hover:text-base-color"
                      onClick={handleReset}
                      disabled={isSubmitting}
                    >
                      Reset
                    </button>
                  )}
                </div>

                {errorInfo && (
                  <div className="space-y-3 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="space-y-1">
                        <p className="font-semibold text-destructive-foreground">{errorInfo.title}</p>
                        <p className="text-sm text-destructive-foreground/90">{errorInfo.message}</p>
                      </div>
                      {errorInfo.details && (
                        <button
                          type="button"
                          onClick={handleCopyDetails}
                          className="inline-flex items-center rounded-md border border-destructive/40 bg-white/10 px-3 py-1 text-xs font-medium text-destructive-foreground"
                        >
                          {copyState === "copied"
                            ? "Copied!"
                            : copyState === "error"
                              ? "Copy failed"
                              : "Copy details"}
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </form>

              <aside className="space-y-6">
                <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
                  <h2 className="text-lg font-semibold text-base-color">CNN Analysis</h2>
                  <p className="text-sm text-muted-foreground">
                    {resultTimestamp
                      ? `Generated ${new Date(resultTimestamp).toLocaleString()}`
                      : "Run the scorer to see results."}
                  </p>

                  {result.error && (
                    <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                      {result.error}
                    </div>
                  )}

                  <div className="mt-6 space-y-4">
                    <div className="rounded-lg border border-border px-4 py-3">
                      <p className="text-xs uppercase tracking-wide text-muted-foreground">Confidence</p>
                      <p className="text-2xl font-semibold text-base-color">{confidencePercent ?? "—"}</p>
                    </div>

                    {ciBounds && (
                      <div className="rounded-lg border border-border px-4 py-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">95% Confidence Interval</p>
                        <p className="text-lg font-semibold text-base-color">{ciBounds}</p>
                      </div>
                    )}

                    {result.monteCarloStats && (
                      <div className="rounded-lg border border-border px-4 py-3 space-y-1">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Monte Carlo Stats</p>
                        <p className="text-sm text-base-color">Samples: {result.monteCarloStats.numSamples}</p>
                        <p className="text-sm text-base-color">Agreement: {(result.monteCarloStats.agreementRate * 100).toFixed(1)}%</p>
                        <p className="text-sm text-base-color">Std Dev: {result.monteCarloStats.stdDev?.toFixed(4)}</p>
                      </div>
                    )}

                    {result.model && (
                      <p className="text-xs text-muted-foreground">Model: {result.model} | Provider: {result.provider}</p>
                    )}
                  </div>
                </div>
              </aside>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 rounded-3xl border border-border bg-[#f5f7fb] p-5 shadow-sm">
              <header className="text-center">
                <h2 className="text-3xl font-semibold text-base-color">Kwiddex Compare App</h2>
                <p className="text-sm text-muted-foreground">Document Comparison Tool</p>
                <p className="mt-1 text-xs text-muted-foreground">ID: {comparisonResult?.analysisId || "ca88a2a7"}</p>
              </header>

              <div className="space-y-2">
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-border bg-white px-3 py-2 text-sm font-medium text-base-color shadow-sm">
                  <FilePlus2 className="h-4 w-4" />
                  Upload questioned
                  <input
                    type="file"
                    accept="image/*,application/pdf,.pdf"
                    onChange={(event) => setQuestionedFile(event.target.files?.[0] ?? null)}
                    className="hidden"
                  />
                </label>
                <div className="rounded-2xl border border-dashed border-border bg-white p-3">
                  <div className="flex items-start gap-3">
                    <span className="rounded-full bg-black/60 px-2 py-1 text-xs font-semibold text-white">{questionedZoom}%</span>
                    <div className="relative flex-1 overflow-hidden rounded-xl border border-border bg-muted/20 p-2">
                      {questionedPreview ? (
                        <img
                          src={questionedPreview}
                          alt="Questioned preview"
                          className="h-52 w-full rounded-lg object-contain"
                          style={{ transform: `scale(${questionedZoom / 100})`, transformOrigin: "top center" }}
                        />
                      ) : (
                        <div className="flex h-52 items-center justify-center text-sm text-muted-foreground">Upload questioned document</div>
                      )}
                    </div>
                    <div className="flex w-12 flex-col items-center gap-3 rounded-xl bg-white px-2 py-3 shadow-sm">
                      <ZoomIn className="h-4 w-4 text-muted-foreground" />
                      <input
                        type="range"
                        min="70"
                        max="200"
                        value={questionedZoom}
                        onChange={(event) => setQuestionedZoom(Number(event.target.value))}
                        className="h-28 [writing-mode:vertical-lr]"
                      />
                      <ZoomOut className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </div>
                </div>
                <p className="text-sm font-semibold text-base-color">Questioned Doc</p>
              </div>

              <div className="space-y-2">
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-border bg-white px-3 py-2 text-sm font-medium text-base-color shadow-sm">
                  <FilePlus2 className="h-4 w-4" />
                  Upload reference
                  <input
                    type="file"
                    accept="image/*,application/pdf,.pdf"
                    onChange={(event) => setKnownFile(event.target.files?.[0] ?? null)}
                    className="hidden"
                  />
                </label>
                <div className="rounded-2xl border border-dashed border-border bg-white p-3">
                  <div className="flex items-start gap-3">
                    <span className="rounded-full bg-black/60 px-2 py-1 text-xs font-semibold text-white">{knownZoom}%</span>
                    <div className="relative flex-1 overflow-hidden rounded-xl border border-border bg-muted/20 p-2">
                      {knownPreview ? (
                        <img
                          src={knownPreview}
                          alt="Reference preview"
                          className="h-52 w-full rounded-lg object-contain"
                          style={{ transform: `scale(${knownZoom / 100})`, transformOrigin: "top center" }}
                        />
                      ) : (
                        <div className="flex h-52 items-center justify-center text-sm text-muted-foreground">Upload reference document</div>
                      )}
                    </div>
                    <div className="flex w-12 flex-col items-center gap-3 rounded-xl bg-white px-2 py-3 shadow-sm">
                      <ZoomIn className="h-4 w-4 text-muted-foreground" />
                      <input
                        type="range"
                        min="70"
                        max="200"
                        value={knownZoom}
                        onChange={(event) => setKnownZoom(Number(event.target.value))}
                        className="h-28 [writing-mode:vertical-lr]"
                      />
                      <ZoomOut className="h-4 w-4 text-muted-foreground" />
                    </div>
                  </div>
                </div>
                <p className="text-sm font-semibold text-base-color">Reference Doc</p>
              </div>

              <textarea
                value={comparisonNotes}
                onChange={(event) => setComparisonNotes(event.target.value)}
                placeholder="Add comments here"
                className="h-24 w-full rounded-xl border border-border bg-white px-4 py-3 text-sm"
              />

              <input
                type="text"
                value={caseName}
                onChange={(event) => setCaseName(event.target.value)}
                placeholder="Enter file name"
                className="w-full rounded-xl border border-border bg-white px-4 py-3 text-sm"
              />

              <button
                type="button"
                onClick={handleRunComparison}
                disabled={isComparing || !knownFile || !questionedFile}
                className="w-full rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-blue-300"
              >
                {isComparing ? "Comparing…" : "Run comparison"}
              </button>

              {comparisonError && (
                <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  <p className="font-semibold text-destructive-foreground">{comparisonError.title}</p>
                  <p className="text-destructive-foreground/90">{comparisonError.message}</p>
                </div>
              )}

              {comparisonResult && (
                <div className="space-y-4 rounded-xl border border-border bg-muted/30 p-4">
                  <div>
                    <p className="text-sm text-muted-foreground">{comparisonResult.summary}</p>
                  </div>

                  <div className="grid gap-3 md:grid-cols-2 text-sm">
                    <div className="rounded-lg border border-border bg-base p-3">
                      <p className="font-semibold">Original — Confidence</p>
                      <p className="text-lg">{typeof comparisonResult.knownResult.confidence === "number" ? `${(comparisonResult.knownResult.confidence * 100).toFixed(1)}%` : "N/A"}</p>
                      {comparisonResult.knownResult.confidenceInterval && (
                        <p className="text-xs text-muted-foreground">CI: {(comparisonResult.knownResult.confidenceInterval.lower * 100).toFixed(1)}% – {(comparisonResult.knownResult.confidenceInterval.upper * 100).toFixed(1)}%</p>
                      )}
                    </div>
                    <div className="rounded-lg border border-border bg-base p-3">
                      <p className="font-semibold">Questioned — Confidence</p>
                      <p className="text-lg">{typeof comparisonResult.questionedResult.confidence === "number" ? `${(comparisonResult.questionedResult.confidence * 100).toFixed(1)}%` : "N/A"}</p>
                      {comparisonResult.questionedResult.confidenceInterval && (
                        <p className="text-xs text-muted-foreground">CI: {(comparisonResult.questionedResult.confidenceInterval.lower * 100).toFixed(1)}% – {(comparisonResult.questionedResult.confidenceInterval.upper * 100).toFixed(1)}%</p>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={handleDownloadComparisonReport}
                    className="inline-flex items-center gap-2 rounded-lg border border-border bg-base px-4 py-2 text-sm font-semibold text-base-color"
                  >
                    Download comparison report
                  </button>
                </div>
              )}

              <div className="rounded-2xl border border-border bg-white px-4 py-3">
                <p className="text-center text-xs font-semibold text-[#9AA2B1]">Entry 1 of 1</p>
                <div className="mt-3 grid grid-cols-6 gap-2 text-xs">
                  <button type="button" onClick={handleResetComparison} className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#2563EB] hover:opacity-80"><FilePlus2 className="h-4 w-4" />New</button>
                  <button type="button" onClick={() => window.print()} className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#344054] hover:opacity-80"><Printer className="h-4 w-4" />Print</button>
                  <button type="button" onClick={handleDownloadComparisonReport} className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#16A34A] hover:opacity-80"><Save className="h-4 w-4" />Save</button>
                  <button type="button" onClick={handleResetComparison} className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#EF4444] hover:opacity-80"><Trash2 className="h-4 w-4" />Delete</button>
                  <button type="button" disabled className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#B7BDC8]"><ChevronLeft className="h-4 w-4" />Prev</button>
                  <button type="button" disabled className="flex flex-col items-center gap-1 border-0 bg-transparent p-0 font-medium text-[#B7BDC8]"><ChevronRight className="h-4 w-4" />Next</button>
                </div>
              </div>

              <div className="text-xs text-muted-foreground">
                {knownFile && <p>Reference: {knownFile.name} {knownFile.size ? `(${formatFileSize(knownFile.size)})` : ""}</p>}
                {questionedFile && <p>Questioned: {questionedFile.name} {questionedFile.size ? `(${formatFileSize(questionedFile.size)})` : ""}</p>}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

Physical.propTypes = {
  initialTab: PropTypes.oneOf(["verification", "comparison"]),
  comparisonOnly: PropTypes.bool,
  embedded: PropTypes.bool,
}
