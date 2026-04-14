import React, { useState } from "react";
import { verifyDocument, API_BASE } from "@/api/verify";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  FileText,
  Shield,
  AlertCircle,
  Hash,
  Wrench,
  Fingerprint,
  ShieldCheck,
  ShieldAlert,
  User,
  Loader2,
  Calendar,
  BarChart3,
  Info,
  Flag,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  Send,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "@/context/AuthContext";
import FileDropZone from "../components/verify/FileDropZone";
import MetadataTable from "../components/verify/MetadataTable";

const KNOWN_EDITORS = {
  "Adobe Acrobat": "Acrobat", "Adobe Photoshop": "Photoshop",
  "Adobe Illustrator": "Illustrator", "Microsoft Word": "Microsoft Word",
  "Microsoft Office": "Microsoft Office", "Apple Preview": "Preview",
  Preview: "Preview", Canva: "Canva", Google: "Google Drive",
  LibreOffice: "LibreOffice", Foxit: "Foxit", TinyWow: "TinyWow",
  Wondershare: "Wondershare", Nitro: "Nitro", Sejda: "Sejda",
};

function VerifyPageContent() {
  const { isAuthenticated, login, user, getToken } = useAuth();
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showMetadata, setShowMetadata] = useState(false);
  const [certLoading, setCertLoading] = useState(false);
  const [certResult, setCertResult] = useState(null);
  const [showReportForm, setShowReportForm] = useState(false);
  const [reportReason, setReportReason] = useState("");
  const [reportSubmitting, setReportSubmitting] = useState(false);
  const [reportError, setReportError] = useState(null);
  const [reportSuccess, setReportSuccess] = useState(false);
  const [showDisputeHistory, setShowDisputeHistory] = useState(false);

  const detectEditor = (metadata) => {
    if (!metadata) return null;
    const fields = [metadata.creator, metadata.producer, metadata.application,
      metadata.CreatorTool, metadata.Producer, metadata.Creator].filter(Boolean);
    for (const field of fields) {
      if (typeof field === "string") {
        for (const [pattern, editor] of Object.entries(KNOWN_EDITORS)) {
          if (field.toLowerCase().includes(pattern.toLowerCase())) return editor;
        }
      }
    }
    return null;
  };

  const verifyCertificate = async (selectedFile) => {
    setCertLoading(true);
    setCertResult(null);
    setShowReportForm(false);
    setReportReason("");
    setReportError(null);
    setReportSuccess(false);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_BASE}/ml/verify-certificate`, { method: "POST", body: formData });
      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(errBody?.detail || "Failed to verify certificate.");
      }
      const data = await response.json();
      setCertResult(data);
    } catch (err) {
      console.error("Certificate verification error:", err);
      setCertResult({ valid: false, has_certificate: false, message: "Could not reach the certificate verification service." });
    }
    setCertLoading(false);
  };

  const submitReport = async () => {
    if (!certResult?.certificate_id) return;
    if (reportReason.trim().length < 50) {
      setReportError("Your reason must be at least 50 characters to ensure a substantive report.");
      return;
    }
    setReportSubmitting(true);
    setReportError(null);
    try {
      const token = await getToken();
      const response = await fetch(`${API_BASE}/ml/report-certificate/${certResult.certificate_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ reason: reportReason.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Failed to submit report.");
      setReportSuccess(true);
      setShowReportForm(false);
      setReportReason("");
      if (file) verifyCertificate(file);
    } catch (err) {
      setReportError(err.message);
    }
    setReportSubmitting(false);
  };

  const handleFileAnalysis = async (selectedFile) => {
    setLoading(true); setResults(null); setError(null); setCertResult(null); setShowMetadata(false);
    try {
      const verificationResult = await verifyDocument(selectedFile);
      const metadata = { ...(verificationResult.core || {}), modificationDate: verificationResult.core?.modDate || null };
      delete metadata.modDate;
      const knownEditor = verificationResult.knownEditor || detectEditor(metadata);
      setResults({ sha256: verificationResult.sha256, metadata: metadata || {}, knownEditor, fileInfo: { name: selectedFile.name, size: selectedFile.size } });
    } catch (err) {
      console.error("Error processing PDF:", err);
      setError(err.message || "Failed to process the PDF file. Please try again.");
    }
    setLoading(false);
    verifyCertificate(selectedFile);
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    if (selectedFile.type !== "application/pdf") { setError("Please select a PDF file only."); return; }
    if (selectedFile.size > 25 * 1024 * 1024) { setError("File size must be less than 25MB."); return; }
    setFile(selectedFile);
    handleFileAnalysis(selectedFile);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024; const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };
  const formatConfidence = (score) => score == null ? "N/A" : `${(score * 100).toFixed(1)}%`;
  const formatDate = (isoString) => {
    if (!isoString) return "N/A";
    try { return new Date(isoString).toLocaleString(); } catch { return isoString; }
  };
  const resetUpload = () => {
    setFile(null); setResults(null); setError(null); setCertResult(null);
    setShowMetadata(false); setShowReportForm(false); setReportReason("");
    setReportError(null); setReportSuccess(false);
  };

  const isOwnCertificate = isAuthenticated && certResult?.reviewer_id &&
    (certResult.reviewer_id === user?.sub || certResult.reviewer_id === user?.email);
  const disputes = certResult?.disputes || [];
  const openDisputes = disputes.filter((d) => d.dispute_status === "open");
  const dismissedDisputes = disputes.filter((d) => d.dispute_status === "dismissed");

  return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
              <ShieldCheck className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-base-color">Verify & Inspect Document</h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a PDF to check if it holds a valid Kwiddex certificate and inspect its metadata.
            You can also report a certificate if you believe it was issued in error.
          </p>
        </div>
        <div className="space-y-6">
          {!file && (
            <Card className="border-2 border-dashed border-border hover:border-blue-300 transition-colors">
              <CardContent className="p-0"><FileDropZone onFileSelect={handleFileSelect} loading={loading} /></CardContent>
            </Card>
          )}
          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
                <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>
              </motion.div>
            )}
          </AnimatePresence>
          {file && (
            <>
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2"><FileText className="w-5 h-5 text-blue-600" /> Uploaded File</CardTitle>
                    <Button variant="outline" size="sm" onClick={resetUpload}>Upload New File</Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <p className="font-mono text-sm bg-surface p-2 rounded break-all">{file.name}</p>
                    <p className="font-mono text-sm bg-surface p-2 rounded">{formatFileSize(file.size)}</p>
                  </div>
                </CardContent>
              </Card>
              <AnimatePresence>
                {certLoading && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <Card><CardContent className="py-8 text-center">
                      <Loader2 className="h-6 w-6 mx-auto animate-spin text-purple-600" />
                      <p className="mt-3 text-muted-foreground">Checking for a Kwiddex certificate...</p>
                    </CardContent></Card>
                  </motion.div>
                )}
                {certResult && !certResult.has_certificate && (
                  <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                    <Card className="border-border"><CardContent className="py-6">
                      <div className="flex items-start gap-3">
                        <Info className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
                        <div>
                          <p className="font-medium text-base-color">This document does not contain a Kwiddex certificate.</p>
                          <p className="text-sm text-muted-foreground mt-1">If you expected one, make sure you are uploading the certified version of the file.</p>
                        </div>
                      </div>
                    </CardContent></Card>
                  </motion.div>
                )}
                {certResult && certResult.has_certificate && (
                  <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="space-y-4">
                    <Card className={certResult.valid ? "border-green-200 bg-green-50 dark:bg-green-950/20 dark:border-green-800" : "border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-800"}>
                      <CardContent className="py-6">
                        <div className="flex items-start gap-3">
                          {certResult.valid ? <ShieldCheck className="w-6 h-6 text-green-600 dark:text-green-400 mt-0.5 shrink-0" /> : <ShieldAlert className="w-6 h-6 text-red-600 dark:text-red-400 mt-0.5 shrink-0" />}
                          <div>
                            <p className={`text-lg font-semibold ${certResult.valid ? "text-green-800 dark:text-green-300" : "text-red-800 dark:text-red-300"}`}>
                              {certResult.valid ? "This document is certified by Kwiddex and has not been modified."
                                : certResult.signature_valid === false ? "This certificate's signature is invalid."
                                : certResult.document_intact === false ? "This document has been modified after certification."
                                : certResult.certificate_active === false ? "This document's certificate has been revoked."
                                : "This certificate could not be verified."}
                            </p>
                            {certResult.valid && certResult.issued_at && <p className="text-sm text-green-700 dark:text-green-400 mt-1">Certified on {formatDate(certResult.issued_at)}.</p>}
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {certResult.has_disputes && openDisputes.length > 0 && (
                      <Card className="border-amber-200 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800">
                        <CardContent className="py-4">
                          <div className="flex items-start gap-3">
                            <Flag className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
                            <div>
                              <p className="font-semibold text-amber-800 dark:text-amber-300">This certificate has {openDisputes.length} open dispute{openDisputes.length > 1 ? "s" : ""}</p>
                              <p className="text-sm text-amber-700 dark:text-amber-400 mt-1">One or more parties have disputed this certification. The certificate remains technically valid. Disputes are informational annotations. Review details below.</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    <Card>
                      <CardHeader className="pb-3"><CardTitle className="flex items-center gap-2 text-base"><Fingerprint className="w-5 h-5 text-purple-600" /> Certificate Details</CardTitle></CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          {certResult.certificate_id && <div className="flex items-start gap-2"><Fingerprint className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" /><div><p className="text-xs text-muted-foreground">Certificate ID</p><p className="text-sm font-mono font-medium">{certResult.certificate_id}</p></div></div>}
                          {certResult.issued_at && <div className="flex items-start gap-2"><Calendar className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" /><div><p className="text-xs text-muted-foreground">Issued</p><p className="text-sm font-medium">{formatDate(certResult.issued_at)}</p></div></div>}
                          {certResult.confidence_score != null && <div className="flex items-start gap-2"><BarChart3 className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" /><div><p className="text-xs text-muted-foreground">CNN Confidence</p><p className="text-sm font-medium">{formatConfidence(certResult.confidence_score)}</p></div></div>}
                          {certResult.reviewer_id && <div className="flex items-start gap-2"><User className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" /><div><p className="text-xs text-muted-foreground">Certified By</p><p className="text-sm font-medium">{certResult.reviewer_id}</p></div></div>}
                        </div>
                        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-border">
                          {certResult.signature_valid != null && <Badge variant="secondary" className={certResult.signature_valid ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"}>Signature {certResult.signature_valid ? "Valid" : "Invalid"}</Badge>}
                          {certResult.document_intact != null && <Badge variant="secondary" className={certResult.document_intact ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"}>Document {certResult.document_intact ? "Intact" : "Modified"}</Badge>}
                          {certResult.certificate_active != null && <Badge variant="secondary" className={certResult.certificate_active ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"}>{certResult.certificate_active ? "Active" : "Revoked"}</Badge>}
                          {certResult.has_disputes && <Badge variant="secondary" className="bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400">{openDisputes.length} Open{dismissedDisputes.length > 0 ? `, ${dismissedDisputes.length} Dismissed` : ""}</Badge>}
                        </div>
                      </CardContent>
                    </Card>

                    {disputes.length > 0 && (
                      <Card>
                        <CardHeader className="pb-3">
                          <button className="flex items-center justify-between w-full" onClick={() => setShowDisputeHistory(!showDisputeHistory)}>
                            <CardTitle className="flex items-center gap-2 text-base"><MessageSquare className="w-5 h-5 text-amber-600" /> Dispute History ({disputes.length})</CardTitle>
                            {showDisputeHistory ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
                          </button>
                        </CardHeader>
                        {showDisputeHistory && (
                          <CardContent className="space-y-4">
                            {disputes.map((dispute, idx) => (
                              <div key={dispute.dispute_id || idx} className={`rounded-lg border p-4 ${dispute.dispute_status === "open" ? "border-amber-200 bg-amber-50/50 dark:border-amber-800 dark:bg-amber-950/10" : dispute.dispute_status === "dismissed" ? "border-border bg-muted/30" : "border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/10"}`}>
                                <div className="flex items-center gap-2 mb-2">
                                  <Badge variant="secondary" className={dispute.dispute_status === "open" ? "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400" : dispute.dispute_status === "dismissed" ? "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300" : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"}>
                                    {dispute.dispute_status === "open" ? "Open" : dispute.dispute_status === "dismissed" ? "Dismissed by Certifier" : "Accepted (Revoked)"}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">Filed {formatDate(dispute.filed_at)}</span>
                                </div>
                                <div className="mb-2">
                                  <p className="text-xs text-muted-foreground mb-1">Reporter: {dispute.reporter_email}</p>
                                  <p className="text-sm text-base-color">{dispute.reason}</p>
                                </div>
                                {dispute.certifier_response && (
                                  <div className="mt-3 pt-3 border-t border-border">
                                    <p className="text-xs text-muted-foreground mb-1">Certifier Response ({formatDate(dispute.resolved_at)})</p>
                                    <p className="text-sm text-base-color italic">{dispute.certifier_response}</p>
                                  </div>
                                )}
                              </div>
                            ))}
                          </CardContent>
                        )}
                      </Card>
                    )}

                    {certResult.valid && !isOwnCertificate && (
                      <Card><CardContent className="py-4">
                        {reportSuccess ? (
                          <div className="flex items-center gap-3 text-green-700 dark:text-green-400"><ShieldCheck className="w-5 h-5 shrink-0" /><p className="text-sm font-medium">Your dispute has been filed and is now visible to anyone who verifies this document.</p></div>
                        ) : !showReportForm ? (
                          <div className="flex items-center justify-between">
                            <div className="flex items-start gap-3">
                              <Flag className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
                              <div>
                                <p className="text-sm font-medium text-base-color">Believe this certification is incorrect?</p>
                                <p className="text-xs text-muted-foreground mt-0.5">File a dispute with a written explanation. Disputes are public, permanent, and visible to anyone who verifies this document.</p>
                              </div>
                            </div>
                            {isAuthenticated ? (
                              <Button variant="outline" size="sm" className="shrink-0 ml-4 border-amber-300 text-amber-700 hover:bg-amber-50 dark:border-amber-700 dark:text-amber-400 dark:hover:bg-amber-950/30" onClick={() => setShowReportForm(true)}><Flag className="w-4 h-4 mr-1" /> Report</Button>
                            ) : (
                              <Button variant="outline" size="sm" className="shrink-0 ml-4" onClick={() => login()}>Sign in to Report</Button>
                            )}
                          </div>
                        ) : (
                          <div className="space-y-3">
                            <div className="flex items-center gap-2"><Flag className="w-5 h-5 text-amber-600" /><p className="text-sm font-semibold text-base-color">File a Dispute</p></div>
                            <p className="text-xs text-muted-foreground">Provide a detailed explanation. Your identity and reason will be permanently attached to this certificate. Minimum 50 characters.</p>
                            <textarea value={reportReason} onChange={(e) => { setReportReason(e.target.value); setReportError(null); }} placeholder="Explain why you believe this certification is incorrect..." className="w-full min-h-[120px] border border-border rounded-lg p-3 text-sm bg-card text-base-color resize-vertical focus:outline-none focus:ring-2 focus:ring-blue-500" />
                            <div className="flex items-center justify-between">
                              <p className={`text-xs ${reportReason.trim().length >= 50 ? "text-green-600" : "text-muted-foreground"}`}>{reportReason.trim().length}/50 characters minimum</p>
                              <div className="flex gap-2">
                                <Button variant="outline" size="sm" onClick={() => { setShowReportForm(false); setReportReason(""); setReportError(null); }}>Cancel</Button>
                                <Button size="sm" className="bg-amber-600 hover:bg-amber-700 text-white" onClick={submitReport} disabled={reportSubmitting || reportReason.trim().length < 50}>
                                  {reportSubmitting ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Send className="w-4 h-4 mr-1" />} Submit Dispute
                                </Button>
                              </div>
                            </div>
                            {reportError && <p className="text-xs text-red-600 dark:text-red-400">{reportError}</p>}
                          </div>
                        )}
                      </CardContent></Card>
                    )}

                    {isOwnCertificate && (
                      <Card className="border-blue-200 dark:border-blue-800"><CardContent className="py-4">
                        <div className="flex items-start gap-3">
                          <Info className="w-5 h-5 text-blue-500 mt-0.5 shrink-0" />
                          <div>
                            <p className="text-sm font-medium text-base-color">You certified this document.</p>
                            <p className="text-xs text-muted-foreground mt-0.5">To revoke this certificate or respond to disputes, visit your certificates dashboard (coming soon).</p>
                          </div>
                        </div>
                      </CardContent></Card>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
              <AnimatePresence>
                {loading && !results && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <Card><CardContent className="py-12 text-center"><Loader2 className="h-8 w-8 mx-auto animate-spin text-blue-600" /><p className="mt-4 text-muted-foreground">Analyzing document...</p></CardContent></Card>
                  </motion.div>
                )}
                {results && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                    {results.knownEditor && (
                      <Card><CardHeader className="pb-3"><CardTitle className="flex items-center gap-2"><Wrench className="w-5 h-5 text-orange-600" /> Editor Detection</CardTitle></CardHeader>
                        <CardContent><Badge variant="secondary" className="bg-orange-100 text-orange-800">{results.knownEditor} Detected</Badge></CardContent></Card>
                    )}
                    {!showMetadata ? (
                      <Button onClick={() => setShowMetadata(true)} variant="outline" className="w-full"><FileText className="w-4 h-4 mr-2" /> View Full Metadata</Button>
                    ) : (
                      <div className="space-y-4">
                        <Card><CardHeader className="pb-3"><CardTitle className="flex items-center gap-2"><Hash className="w-5 h-5 text-green-600" /> File Integrity Hash</CardTitle></CardHeader>
                          <CardContent><div className="font-mono text-sm bg-surface p-3 rounded break-all">{results.sha256}</div></CardContent></Card>
                        <MetadataTable metadata={results.metadata} />
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function VerifyPage() {
  return <VerifyPageContent />;
}
