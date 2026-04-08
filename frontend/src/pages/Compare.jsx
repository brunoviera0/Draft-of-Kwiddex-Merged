import React, { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Upload,
  Plus,
  Trash2,
  Download,
  Scale,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Newspaper,
  Copy,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import FileDropZone from "../components/verify/FileDropZone";

import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfjsWorker;

// =====================================================
// LWSP ENGINE (Linear Wave Stochastic Process)
// Spectral similarity via 2D FFT cross-correlation
// =====================================================

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function fft1D(re, im) {
  const N = re.length;
  if (N <= 1) return;
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang),
      wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curRe = 1,
        curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k],
          uIm = im[i + k];
        const vRe =
          re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm =
          re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

function compute2DFFT(gray, W, H, size) {
  const re = new Float32Array(size * size);
  const im = new Float32Array(size * size);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) re[y * size + x] = gray[y * W + x];
  const rowRe = new Float32Array(size);
  const rowIm = new Float32Array(size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      rowRe[x] = re[y * size + x];
      rowIm[x] = im[y * size + x];
    }
    fft1D(rowRe, rowIm);
    for (let x = 0; x < size; x++) {
      re[y * size + x] = rowRe[x];
      im[y * size + x] = rowIm[x];
    }
  }
  const colRe = new Float32Array(size);
  const colIm = new Float32Array(size);
  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      colRe[y] = re[y * size + x];
      colIm[y] = im[y * size + x];
    }
    fft1D(colRe, colIm);
    for (let y = 0; y < size; y++) {
      re[y * size + x] = colRe[y];
      im[y * size + x] = colIm[y];
    }
  }
  return { re, im, size };
}

function computePSD(fft, size) {
  const N = size * size;
  const psd = new Float32Array(N);
  for (let i = 0; i < N; i++)
    psd[i] = (fft.re[i] * fft.re[i] + fft.im[i] * fft.im[i]) / N;
  let norm = 0;
  for (let i = 0; i < N; i++) norm += psd[i] * psd[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < N; i++) psd[i] /= norm;
  return psd;
}

function gaussianBlur(data, W, H, sigma) {
  const out = new Float32Array(data.length);
  const radius = Math.ceil(sigma * 2);
  const kernel = [];
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(v);
    ksum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;
  const tmp = new Float32Array(data.length);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let val = 0;
      for (let k = -radius; k <= radius; k++) {
        const xi = Math.max(0, Math.min(W - 1, x + k));
        val += data[y * W + xi] * kernel[k + radius];
      }
      tmp[y * W + x] = val;
    }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let val = 0;
      for (let k = -radius; k <= radius; k++) {
        const yi = Math.max(0, Math.min(H - 1, y + k));
        val += tmp[yi * W + x] * kernel[k + radius];
      }
      out[y * W + x] = val;
    }
  return out;
}

function toGrayscaleFiltered(pixData, W, H, highCut) {
  const result = new Float32Array(W * H);
  const data = pixData.data;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = (y * pixData.width + x) * 4;
      result[y * W + x] =
        0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    }
  let mean = 0;
  for (let i = 0; i < result.length; i++) mean += result[i];
  mean /= result.length;
  for (let i = 0; i < result.length; i++) result[i] -= mean;
  const smoothed = gaussianBlur(result, W, H, highCut);
  for (let i = 0; i < result.length; i++) result[i] -= smoothed[i] * 0.15;
  return result;
}

function runLWSP(canvasQ, canvasR, highCut) {
  const ctxQ = canvasQ.getContext("2d");
  const ctxR = canvasR.getContext("2d");
  const pixQ = ctxQ.getImageData(0, 0, canvasQ.width, canvasQ.height);
  const pixR = ctxR.getImageData(0, 0, canvasR.width, canvasR.height);
  const W = Math.min(pixQ.width, pixR.width);
  const H = Math.min(pixQ.height, pixR.height);
  const grayQ = toGrayscaleFiltered(pixQ, W, H, highCut);
  const grayR = toGrayscaleFiltered(pixR, W, H, highCut);
  const size = nextPow2(Math.max(W, H));
  const fftQ = compute2DFFT(grayQ, W, H, size);
  const fftR = compute2DFFT(grayR, W, H, size);
  const psdQ = computePSD(fftQ, size);
  const psdR = computePSD(fftR, size);
  let dotProduct = 0,
    sumQ2 = 0,
    sumR2 = 0;
  for (let i = 0; i < psdQ.length; i++) {
    dotProduct += psdQ[i] * psdR[i];
    sumQ2 += psdQ[i] * psdQ[i];
    sumR2 += psdR[i] * psdR[i];
  }
  const denom = Math.sqrt(sumQ2 * sumR2);
  if (denom === 0) return 0;
  return Math.max(0, Math.min(100, (dotProduct / denom) * 100));
}

// =====================================================
// IMAGE VIEWER COMPONENT
// =====================================================

const SCALE_MIN = 0.2;
const SCALE_MAX = 8;

function sliderToScale(v) {
  return SCALE_MIN * Math.pow(SCALE_MAX / SCALE_MIN, parseFloat(v) / 100);
}

function scaleToSlider(s) {
  return Math.round(
    (100 * Math.log(Math.max(s, SCALE_MIN) / SCALE_MIN)) /
      Math.log(SCALE_MAX / SCALE_MIN)
  );
}

function ImageViewer({
  label,
  image,
  state,
  onStateChange,
  canvasRef,
  onFileSelect,
}) {
  const wrapRef = useRef(null);
  const dragRef = useRef({ dragging: false, lastX: 0, lastY: 0 });
  const touchRef = useRef({ lastTouches: [], lastDist: 0 });
  const fileRef = useRef(null);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) return;
    ctx.save();
    ctx.translate(state.ox, state.oy);
    ctx.scale(state.scale, state.scale);
    ctx.drawImage(image, 0, 0);
    ctx.restore();
  }, [image, state, canvasRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    render();
  }, [render, canvasRef]);

  useEffect(() => {
    render();
  }, [render]);

  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const wrap = wrapRef.current;
      if (!canvas || !wrap) return;
      canvas.width = wrap.clientWidth;
      canvas.height = wrap.clientHeight;
      render();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [render, canvasRef]);

  const handleMouseDown = (e) => {
    dragRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY };
    canvasRef.current.style.cursor = "grabbing";
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!dragRef.current.dragging) return;
      const dx = e.clientX - dragRef.current.lastX;
      const dy = e.clientY - dragRef.current.lastY;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
      onStateChange((prev) => ({
        ...prev,
        ox: prev.ox + dx,
        oy: prev.oy + dy,
      }));
    };
    const handleMouseUp = () => {
      dragRef.current.dragging = false;
      if (canvasRef.current) canvasRef.current.style.cursor = "grab";
    };
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [onStateChange, canvasRef]);

  const handleWheel = (e) => {
    e.preventDefault();
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.95 : 1.05;
    onStateChange((prev) => {
      const newScale = Math.min(
        SCALE_MAX,
        Math.max(SCALE_MIN, prev.scale * delta)
      );
      const ratio = newScale / prev.scale;
      return {
        scale: newScale,
        ox: cx - ratio * (cx - prev.ox),
        oy: cy - ratio * (cy - prev.oy),
      };
    });
  };

  const getTouchDist = (touches) => {
    const dx = touches[0].clientX - touches[1].clientX;
    const dy = touches[0].clientY - touches[1].clientY;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const handleTouchStart = (e) => {
    e.preventDefault();
    touchRef.current.lastTouches = Array.from(e.touches);
    if (e.touches.length === 2)
      touchRef.current.lastDist = getTouchDist(e.touches);
  };

  const handleTouchMove = (e) => {
    e.preventDefault();
    const touches = Array.from(e.touches);
    const last = touchRef.current.lastTouches;
    if (touches.length === 1 && last.length >= 1) {
      const dx = touches[0].clientX - last[0].clientX;
      const dy = touches[0].clientY - last[0].clientY;
      onStateChange((prev) => ({
        ...prev,
        ox: prev.ox + dx,
        oy: prev.oy + dy,
      }));
    } else if (touches.length === 2) {
      const dist = getTouchDist(touches);
      if (touchRef.current.lastDist > 0) {
        const ratio = dist / touchRef.current.lastDist;
        const midX = (touches[0].clientX + touches[1].clientX) / 2;
        const midY = (touches[0].clientY + touches[1].clientY) / 2;
        const rect = canvasRef.current.getBoundingClientRect();
        const cx = midX - rect.left;
        const cy = midY - rect.top;
        const clampedRatio = Math.min(1.06, Math.max(0.94, ratio));
        onStateChange((prev) => {
          const newScale = Math.min(
            SCALE_MAX,
            Math.max(SCALE_MIN, prev.scale * clampedRatio)
          );
          const scaleRatio = newScale / prev.scale;
          return {
            scale: newScale,
            ox: cx - scaleRatio * (cx - prev.ox),
            oy: cy - scaleRatio * (cy - prev.oy),
          };
        });
      }
      touchRef.current.lastDist = dist;
    }
    touchRef.current.lastTouches = touches;
  };

  const handleTouchEnd = (e) => {
    touchRef.current.lastTouches = Array.from(e.touches);
    if (e.touches.length < 2) touchRef.current.lastDist = 0;
  };

  const handleSlider = (e) => {
    const newScale = sliderToScale(e.target.value);
    const canvas = canvasRef.current;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    onStateChange((prev) => {
      const ratio = newScale / prev.scale;
      return {
        scale: newScale,
        ox: cx - ratio * (cx - prev.ox),
        oy: cy - ratio * (cy - prev.oy),
      };
    });
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const scaleX = canvas.width / img.width;
        const scaleY = canvas.height / img.height;
        const fitScale = Math.min(scaleX, scaleY);
        onStateChange({
          scale: fitScale,
          ox: (canvas.width - img.width * fitScale) / 2,
          oy: (canvas.height - img.height * fitScale) / 2,
        });
        onFileSelect(img);
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  return (
    <div className="space-y-1">
      <div
        ref={wrapRef}
        className="relative w-full bg-card border border-border rounded-lg overflow-hidden"
        style={{ height: 280, touchAction: "none" }}
      >
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
          style={{ cursor: image ? "grab" : "default" }}
          onMouseDown={handleMouseDown}
          onWheel={handleWheel}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
        />
        {!image && (
          <div className="absolute inset-0 flex items-center justify-center">
            <button
              onClick={() => fileRef.current?.click()}
              className="flex flex-col items-center gap-2 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              <Upload className="w-8 h-8" />
              <span className="text-sm">Upload image</span>
            </button>
          </div>
        )}
        {image && (
          <div className="absolute right-2 top-2 bottom-2 flex items-center">
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={scaleToSlider(state.scale)}
              onChange={handleSlider}
              className="h-full opacity-30 hover:opacity-70 transition-opacity"
              style={{
                writingMode: "vertical-lr",
                direction: "rtl",
                width: 18,
                WebkitAppearance: "slider-vertical",
                cursor: "pointer",
              }}
            />
          </div>
        )}
      </div>
      <div className="flex items-center justify-between px-1">
        <span className="text-sm font-semibold text-base-color">{label}</span>
        {image && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fileRef.current?.click()}
            className="h-7 px-2 text-xs text-muted-foreground"
          >
            Replace
          </Button>
        )}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFile}
      />
    </div>
  );
}


// =====================================================
// OCR TAB COMPONENT
// =====================================================

function OcrTab() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [extractedText, setExtractedText] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = async (selectedFile) => {
    if (!selectedFile) return;
    if (selectedFile.type !== "application/pdf") {
      setError("Please select a PDF file only.");
      return;
    }
    if (selectedFile.size > 25 * 1024 * 1024) {
      setError("File size must be less than 25MB.");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setExtractedText(null);
    setLoading(true);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const loadingTask = getDocument({
        data: new Uint8Array(arrayBuffer),
        useSystemFonts: true,
      });
      const pdf = await loadingTask.promise;
      let textContent = "";

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const textObj = await page.getTextContent();
        const pageText = textObj.items
          .map((item) => ("str" in item ? item.str : ""))
          .join(" ")
          .trim();
        if (pageText) textContent += pageText + "\n\n";
      }

      if (textContent.trim().length === 0) {
        setExtractedText("No extractable text found. This PDF may contain only images or scanned content.");
      } else {
        setExtractedText(textContent.trim());
      }
    } catch (err) {
      console.error("OCR Error:", err);
      setError("Could not read this PDF. It may be encrypted, corrupted, or in an unsupported format. Error: " + (err.message || "Unknown"));
    }
    setLoading(false);
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const reset = () => {
    setFile(null);
    setExtractedText(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </motion.div>
        )}
      </AnimatePresence>

      {extractedText === null ? (
        <Card className="border-2 border-dashed border-border hover:border-amber-300 transition-colors">
          <CardContent className="p-0">
            <FileDropZone
              onFileSelect={handleFileSelect}
              loading={loading}
              title="Upload PDF for Text Extraction"
              description="Drag and drop a PDF here, or click to browse"
              acceptLabel="Choose PDF File"
            />
          </CardContent>
        </Card>
      ) : (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    Text Extracted
                  </CardTitle>
                  <Button variant="outline" size="sm" onClick={reset}>
                    Extract Another
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-3">
                  Extracted from{" "}
                  <span className="font-medium text-base-color">
                    {file.name}
                  </span>
                </p>
                <div className="relative">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => copyToClipboard(extractedText)}
                    className="absolute top-2 right-2 z-10"
                  >
                    {copySuccess ? (
                      <>
                        <CheckCircle className="w-3 h-3 mr-1 text-green-500" />{" "}
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3 mr-1" /> Copy
                      </>
                    )}
                  </Button>
                  <pre className="w-full h-80 overflow-y-auto bg-surface p-4 rounded-md whitespace-pre-wrap font-sans text-sm text-base-color">
                    {extractedText}
                  </pre>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  );
}

// =====================================================
// COMPARE TAB COMPONENT
// =====================================================

function createBlankEntry() {
  return {
    imgQ: null,
    imgR: null,
    stateQ: { scale: 1, ox: 0, oy: 0 },
    stateR: { scale: 1, ox: 0, oy: 0 },
    notes: "",
    filename: "",
    score: null,
  };
}

function CompareTab() {
  const [entries, setEntries] = useState([createBlankEntry()]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [computing, setComputing] = useState(false);
  const [highCut, setHighCut] = useState(1.5);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const canvasQRef = useRef(null);
  const canvasRRef = useRef(null);

  const entry = entries[currentIdx];

  const updateEntry = (field, value) => {
    setEntries((prev) => {
      const next = [...prev];
      next[currentIdx] = { ...next[currentIdx], [field]: value };
      return next;
    });
  };

  const handleCompare = () => {
    if (!entry.imgQ || !entry.imgR) return;
    setComputing(true);
    setTimeout(() => {
      try {
        const score = runLWSP(canvasQRef.current, canvasRRef.current, highCut);
        updateEntry("score", score.toFixed(1));
      } catch (err) {
        console.error("Comparison failed:", err);
      }
      setComputing(false);
    }, 50);
  };

  const addEntry = () => {
    setEntries((prev) => [...prev, createBlankEntry()]);
    setCurrentIdx(entries.length);
  };

  const deleteEntry = () => {
    setShowDeleteModal(false);
    if (entries.length === 1) {
      setEntries([createBlankEntry()]);
      setCurrentIdx(0);
    } else {
      setEntries((prev) => {
        const next = [...prev];
        next.splice(currentIdx, 1);
        return next;
      });
      setCurrentIdx((prev) => Math.min(prev, entries.length - 2));
    }
  };

  const [downloadError, setDownloadError] = useState(null);

  const downloadEntry = () => {
    setDownloadError(null);

    if (!entry.imgQ || !entry.imgR) {
      setDownloadError("Please upload both a questioned and reference document before downloading.");
      return;
    }
    if (!entry.filename.trim()) {
      setDownloadError("Please enter a case or file name before downloading.");
      return;
    }

    const canvas = document.createElement("canvas");
    const totalW = 500;
    const totalH = 780;
    canvas.width = totalW;
    canvas.height = totalH;
    const ctx = canvas.getContext("2d");

    // Background
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, totalW, totalH);

    // Header
    ctx.fillStyle = "#d97706";
    ctx.fillRect(0, 0, totalW, 56);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 20px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("⚖️ Scales of Justice — LWSP Analysis", totalW / 2, 36);

    // Case name
    ctx.fillStyle = "#ccc";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Case: " + entry.filename, 16, 78);
    ctx.fillText("Date: " + new Date().toLocaleString(), 16, 96);

    let y = 116;
    [canvasQRef, canvasRRef].forEach((ref, i) => {
      const label = i === 0 ? "Questioned Document" : "Reference Document";
      ctx.fillStyle = "#fff";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(label, 16, y);
      y += 6;

      // Draw border
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 1;
      ctx.strokeRect(14, y, totalW - 28, 220);

      if (ref.current) {
        ctx.drawImage(ref.current, 15, y + 1, totalW - 30, 218);
      }
      y += 230;
    });

    // Score
    ctx.fillStyle = "#d97706";
    ctx.font = "bold 28px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "Similarity Score: " + (entry.score != null ? entry.score : "Not computed"),
      totalW / 2,
      y + 10
    );
    y += 30;

    // Notes
    if (entry.notes.trim()) {
      ctx.fillStyle = "#ccc";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "left";
      const lines = entry.notes.split("\n");
      ctx.fillText("Notes:", 16, y + 10);
      y += 24;
      for (const line of lines.slice(0, 6)) {
        ctx.fillText(line.slice(0, 80), 16, y);
        y += 16;
      }
    }

    // Footer
    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Generated by Kwiddex — kwiddex.com", totalW / 2, totalH - 12);

    const link = document.createElement("a");
    link.download = entry.filename.trim().replace(/\s+/g, "_") + "_report.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  return (
    <div className="space-y-4">
      {/* Image panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ImageViewer
          label="Questioned Document"
          image={entry.imgQ}
          state={entry.stateQ}
          onStateChange={(updater) => {
            setEntries((prev) => {
              const next = [...prev];
              const newState =
                typeof updater === "function"
                  ? updater(next[currentIdx].stateQ)
                  : updater;
              next[currentIdx] = { ...next[currentIdx], stateQ: newState };
              return next;
            });
          }}
          canvasRef={canvasQRef}
          onFileSelect={(img) => {
            updateEntry("imgQ", img);
            updateEntry("score", null);
          }}
        />
        <ImageViewer
          label="Reference Document"
          image={entry.imgR}
          state={entry.stateR}
          onStateChange={(updater) => {
            setEntries((prev) => {
              const next = [...prev];
              const newState =
                typeof updater === "function"
                  ? updater(next[currentIdx].stateR)
                  : updater;
              next[currentIdx] = { ...next[currentIdx], stateR: newState };
              return next;
            });
          }}
          canvasRef={canvasRRef}
          onFileSelect={(img) => {
            updateEntry("imgR", img);
            updateEntry("score", null);
          }}
        />
      </div>

      {/* Analysis controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <span className="text-lg">⚖️</span>
            LWSP Spectral Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">
                  High-Cut Filter (sigma)
                </span>
                <span className="font-mono font-medium">
                  {highCut.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min="0.5"
                max="5.0"
                step="0.1"
                value={highCut}
                onChange={(e) => setHighCut(parseFloat(e.target.value))}
                className="w-full accent-amber-600"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>More detail (noise)</span>
                <span>Smoother (less detail)</span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Button
                onClick={handleCompare}
                disabled={!entry.imgQ || !entry.imgR || computing}
                className="bg-amber-600 hover:bg-amber-700"
              >
                {computing ? (
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Scale className="w-4 h-4 mr-2" />
                )}
                {computing ? "Computing..." : "Run Comparison"}
              </Button>

              {entry.score != null && (
                <div>
                  <p className="text-xs text-muted-foreground">
                    Similarity Score
                  </p>
                  <p className="text-2xl font-bold font-mono">{entry.score}</p>
                </div>
              )}
            </div>

            {(!entry.imgQ || !entry.imgR) && (
              <p className="text-xs text-muted-foreground">
                Upload images to both panels before comparing.
              </p>
            )}

              <div className="text-xs text-muted-foreground space-y-2 border-t border-border pt-3">
                <p className="font-medium text-base-color">How it works</p>
                <p>
                  This tool calculates a similarity score between two document images using the Linear Wave Stochastic Process (LWSP) method. Each image is converted to grayscale and passed through a band-pass filter to isolate meaningful visual features. Both filtered images are then transformed into the frequency domain using a 2D Fast Fourier Transform (FFT). The resulting Power Spectral Densities (PSDs) are compared via normalized cross-correlation to produce a similarity score from 0 to 100.
                </p>
                <p>
                  The <span className="font-medium">high-cut filter (sigma)</span> controls how much fine detail is included in the comparison. A lower value preserves more detail but also more noise. A higher value smooths the image, focusing on larger structural features. Adjust it based on what you're comparing — fine print and signatures benefit from lower values, while overall page layout comparisons work better with higher values.
                </p>
              </div>
          </div>
        </CardContent>
      </Card>


{/* Notes + Case Name + Download */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="text-sm font-medium text-muted-foreground block mb-1">
            Notes (optional)
          </label>
          <textarea
            value={entry.notes}
            onChange={(e) => updateEntry("notes", e.target.value)}
            placeholder="Add observations or comments..."
            className="w-full min-h-[80px] border border-border rounded-lg p-3 text-sm bg-card text-base-color resize-vertical"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-muted-foreground block mb-1">
            Case / File Name
          </label>
          <input
            type="text"
            value={entry.filename}
            onChange={(e) => updateEntry("filename", e.target.value)}
            placeholder="Enter case or file name..."
            className="w-full border border-border rounded-lg p-3 text-sm bg-card text-base-color"
          />
        </div>
        <div className="flex flex-col justify-end">
          <Button
            onClick={downloadEntry}
            className="bg-amber-600 hover:bg-amber-700 w-full"
          >
            <Download className="w-4 h-4 mr-2" />
            Download Report
          </Button>
          {downloadError && (
            <p className="text-xs text-red-500 mt-1">{downloadError}</p>
          )}
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between py-2">
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={addEntry}>
            <Plus className="w-4 h-4 mr-1" /> New
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowDeleteModal(true)}
          >
            <Trash2 className="w-4 h-4 mr-1" /> Delete
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
            disabled={currentIdx <= 0}
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            {currentIdx + 1} / {entries.length}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setCurrentIdx((i) => Math.min(entries.length - 1, i + 1))
            }
            disabled={currentIdx >= entries.length - 1}
          >
            <ChevronRight className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Delete modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-card border border-border rounded-xl p-6 max-w-sm w-full mx-4 space-y-4">
            <h3 className="text-lg font-semibold">Delete Entry</h3>
            <p className="text-sm text-muted-foreground">
              Are you sure? This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <Button
                variant="outline"
                onClick={() => setShowDeleteModal(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={deleteEntry}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// =====================================================
// MAIN COMPARE PAGE WITH TABS
// =====================================================

export default function ComparePage() {
  const [activeTab, setActiveTab] = useState("compare");

  return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-12 h-12 bg-amber-600 rounded-xl flex items-center justify-center text-2xl">
              ⚖️
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-base-color">
              Scales of Justice
            </h1>

          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            {activeTab === "compare"
              ? "Upload two document images to calculate their similarity using spectral frequency analysis."
              : "Upload a PDF to extract its embedded text content."}
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 mb-6 bg-card border border-border rounded-lg p-1 max-w-xs mx-auto">
          <button
            onClick={() => setActiveTab("compare")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === "compare"
                ? "bg-amber-600 text-white"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Scale className="w-4 h-4" />
            Compare
          </button>
          <button
            onClick={() => setActiveTab("ocr")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === "ocr"
                ? "bg-amber-600 text-white"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Newspaper className="w-4 h-4" />
            Extract Text
          </button>
        </div>

        {/* Tab content */}
        {activeTab === "compare" ? <CompareTab /> : <OcrTab />}
      </div>
    </div>
  );
}
