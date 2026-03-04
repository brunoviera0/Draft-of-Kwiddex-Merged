import { PDFDocument, StandardFonts, rgb } from "pdf-lib"
import type { AiResult } from "./openaiScorer"
import { createQrMatrix } from "./qr"

export type ReportPayload = {
  analysisId: string
  createdAt: number
  branch: "image" | "pdf"
  file: {
    name: string
    mimetype: string
    size: number
  }
  result: AiResult & {
    elapsedMs: number
  }
  verificationCode: string
}

const PAGE_WIDTH = 612
const PAGE_HEIGHT = 792
const MARGIN = 48
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN * 2

export async function buildExplanationReport(payload: ReportPayload): Promise<Uint8Array> {
  const pdf = await PDFDocument.create()
  const page = pdf.addPage([PAGE_WIDTH, PAGE_HEIGHT])
  const bodyFont = await pdf.embedFont(StandardFonts.Helvetica)
  const boldFont = await pdf.embedFont(StandardFonts.HelveticaBold)

  let cursorY = PAGE_HEIGHT - MARGIN

  const moveCursor = (amount: number) => {
    cursorY -= amount
    if (cursorY < MARGIN) {
      cursorY = MARGIN
    }
  }

  const drawParagraph = (
    text: string,
    options: { size?: number; font?: any; color?: { r: number; g: number; b: number }; spacing?: number } = {}
  ) => {
    const { size = 12, font = bodyFont, color = rgb(0.16, 0.17, 0.2), spacing = 4 } = options
    const lines = wrapText(text, font, size, CONTENT_WIDTH)
    for (const line of lines) {
      page.drawText(line, { x: MARGIN, y: cursorY - size, size, font, color })
      moveCursor(size + spacing)
    }
  }

  const drawHeading = (title: string) => {
    drawParagraph(title, { font: boldFont, size: 14, spacing: 10, color: rgb(0.12, 0.13, 0.2) })
  }

  const drawBulletList = (items: string[]) => {
    if (!items || items.length === 0) {
      drawParagraph("None reported.", { size: 11, color: rgb(0.45, 0.48, 0.52) })
      return
    }

    const bulletIndent = 12
    const textIndent = 18
    const size = 11
    for (const item of items) {
      const lines = wrapText(item, bodyFont, size, CONTENT_WIDTH - textIndent)
      let firstLine = true
      for (const line of lines) {
        if (firstLine) {
          page.drawText("•", {
            x: MARGIN,
            y: cursorY - size,
            size,
            font: bodyFont,
            color: rgb(0.12, 0.13, 0.2),
          })
        }
        page.drawText(line, {
          x: MARGIN + (firstLine ? bulletIndent : textIndent),
          y: cursorY - size,
          size,
          font: bodyFont,
          color: rgb(0.16, 0.17, 0.2),
        })
        moveCursor(size + 4)
        firstLine = false
      }
    }
  }

  // Header
  drawParagraph("KwiddeX Intelligence Physical Document Authenticity (beta)", {
    font: boldFont,
    size: 20,
    spacing: 12,
    color: rgb(0.07, 0.34, 0.73),
  })
  drawParagraph(`Generated: ${new Date(payload.createdAt).toLocaleString()}`, {
    size: 10,
    color: rgb(0.41, 0.44, 0.48),
    spacing: 12,
  })

  // Summary block
  const summaryTop = cursorY
  const summaryHeight = 100
  page.drawRectangle({
    x: MARGIN,
    y: summaryTop - summaryHeight,
    width: CONTENT_WIDTH,
    height: summaryHeight,
    color: rgb(0.96, 0.97, 1),
    borderColor: rgb(0.78, 0.85, 0.93),
    borderWidth: 1,
  })

  const overallScore = `${Math.round(payload.result.score)} / 100`
  page.drawText("Overall score", {
    x: MARGIN + 16,
    y: summaryTop - 28,
    size: 10,
    font: bodyFont,
    color: rgb(0.41, 0.44, 0.48),
  })
  page.drawText(overallScore, {
    x: MARGIN + 16,
    y: summaryTop - 52,
    size: 24,
    font: boldFont,
    color: rgb(0.07, 0.34, 0.73),
  })

  const confidenceLabel = deriveConfidenceLabel(payload.result)
  page.drawText("Confidence", {
    x: MARGIN + 210,
    y: summaryTop - 28,
    size: 10,
    font: bodyFont,
    color: rgb(0.41, 0.44, 0.48),
  })
  page.drawText(confidenceLabel, {
    x: MARGIN + 210,
    y: summaryTop - 50,
    size: 18,
    font: boldFont,
    color: rgb(0.12, 0.6, 0.48),
  })

  const summaryLines = [
    `Analysis ID: ${payload.analysisId}`,
    `Model: ${payload.result.model}`,
    `Elapsed: ${payload.result.elapsedMs}ms`,
    `Document type: ${payload.branch.toUpperCase()}`,
  ]
  let metaY = summaryTop - 28
  for (const line of summaryLines) {
    page.drawText(line, {
      x: MARGIN + 380,
      y: metaY,
      size: 10,
      font: bodyFont,
      color: rgb(0.25, 0.28, 0.32),
    })
    metaY -= 16
  }

  cursorY = summaryTop - summaryHeight - 20

  // Document overview
  drawHeading("Document overview")
  drawParagraph(`Filename: ${payload.file.name}`, { size: 11 })
  drawParagraph(`Type: ${payload.file.mimetype}`, { size: 11 })
  drawParagraph(`Size: ${formatBytes(payload.file.size)}`, { size: 11, spacing: 12 })

  if (payload.result.qualitySignals?.previewDataUrl) {
    try {
      const { bytes, mime } = decodeDataUrl(payload.result.qualitySignals.previewDataUrl)
      const embedded =
        mime === "image/png" ? await pdf.embedPng(bytes) : await pdf.embedJpg(bytes)
      const maxPreviewWidth = 200
      const scaled = embedded.scale(maxPreviewWidth / embedded.width)
      page.drawImage(embedded, {
        x: MARGIN,
        y: cursorY - scaled.height - 10,
        width: scaled.width,
        height: scaled.height,
      })
      cursorY -= scaled.height + 20
    } catch (error) {
      console.warn("[report] failed to embed preview", error)
    }
  }

  // Subscores
  drawHeading("Subscores")
  const subscoreEntries = Object.entries(payload.result.subscores ?? {})
  if (subscoreEntries.length === 0) {
    drawParagraph("No subscores reported.", { size: 11, color: rgb(0.45, 0.48, 0.52) })
  } else {
    for (const [label, value] of subscoreEntries) {
      drawParagraph(`${label}: ${Math.round(value)} / 100`, { size: 11 })
    }
  }

  // Reasons, flags, suggestions
  drawHeading("Reasons")
  drawBulletList(payload.result.reasons)

  drawHeading("Flags")
  drawBulletList(payload.result.flags)

  drawHeading("Suggestions")
  drawBulletList(payload.result.suggestions)

  // Quality notes
  drawHeading("Quality notes")
  drawBulletList(buildQualityNotes(payload.result))

  drawParagraph("Disclaimer: Automated estimate; not forensic proof. Image quality affects reliability.", {
    size: 9,
    color: rgb(0.45, 0.48, 0.52),
    spacing: 12,
  })

  const verificationUrl = buildVerificationUrl(payload.verificationCode)
  const qrValue = buildVerificationValue(payload.verificationCode)
  const qrMatrix = createQrMatrix(qrValue)
  const qrSize = 90
  const moduleSize = qrSize / qrMatrix.size
  const qrLeft = PAGE_WIDTH - MARGIN - qrSize
  const qrBottom = MARGIN

  for (let row = 0; row < qrMatrix.size; row += 1) {
    for (let col = 0; col < qrMatrix.size; col += 1) {
      if (!qrMatrix.modules[row][col]) continue
      page.drawRectangle({
        x: qrLeft + col * moduleSize,
        y: qrBottom + (qrMatrix.size - 1 - row) * moduleSize,
        width: moduleSize,
        height: moduleSize,
        color: rgb(0.06, 0.07, 0.09),
      })
    }
  }

  page.drawText("Scan to verify", {
    x: PAGE_WIDTH - MARGIN - 95,
    y: qrBottom + qrSize + 6,
    size: 10,
    font: boldFont,
    color: rgb(0.16, 0.17, 0.2),
  })

  page.drawText(verificationUrl, {
    x: MARGIN,
    y: MARGIN + 20,
    size: 9,
    font: bodyFont,
    color: rgb(0.07, 0.34, 0.73),
  })

  return pdf.save()
}

function wrapText(text: string, font: any, size: number, maxWidth: number) {
  const words = text.split(/\s+/)
  const lines: string[] = []
  let current = ""
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word
    if (font.widthOfTextAtSize(candidate, size) > maxWidth && current) {
      lines.push(current)
      current = word
    } else {
      current = candidate
    }
  }
  if (current) {
    lines.push(current)
  }
  return lines
}

function deriveConfidenceLabel(result: AiResult & { elapsedMs: number }) {
  if (typeof result.confidence === "number") {
    if (result.confidence >= 0.75) return "High"
    if (result.confidence >= 0.4) return "Medium"
    return "Low"
  }
  if (result.score >= 80) return "High"
  if (result.score >= 50) return "Medium"
  return "Low"
}

function buildQualityNotes(result: AiResult) {
  const notes: string[] = []
  const subs = result.subscores ?? {}
  if (typeof subs.quality_blur === "number") {
    notes.push(`Sharpness: ${Math.round(subs.quality_blur)} / 100`)
  }
  if (typeof subs.quality_glare === "number") {
    notes.push(`Glare handling: ${Math.round(subs.quality_glare)} / 100`)
  }
  if (typeof subs.quality_skew === "number") {
    notes.push(`Alignment: ${Math.round(subs.quality_skew)} / 100`)
  }
  if (notes.length === 0) {
    notes.push("No local quality subscores were returned.")
  }
  return notes
}

function buildVerificationUrl(code: string) {
  const base = process.env.VERIFICATION_BASE_URL || "https://kwiddex.local"
  return `${base}/physical/result/${code}`
}

function buildVerificationValue(code: string) {
  return buildVerificationUrl(code)
}

function decodeDataUrl(dataUrl: string): { bytes: Uint8Array; mime: string } {
  const match = /^data:(?<mime>[^;]+);base64,(?<data>[\s\S]+)$/i.exec(dataUrl)
  if (!match || !match.groups) {
    throw new Error("Invalid data URL")
  }
  const { mime, data } = match.groups as { mime: string; data: string }
  return { bytes: Buffer.from(data, "base64"), mime }
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes)) return "Unknown"
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }
  return `${bytes} B`
}
