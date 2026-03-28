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
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import FileDropZone from "../components/verify/FileDropZone";
import MetadataTable from "../components/verify/MetadataTable";

const KNOWN_EDITORS = {
  "Adobe Acrobat": "Acrobat",
  "Adobe Photoshop": "Photoshop",
  "Adobe Illustrator": "Illustrator",
  "Microsoft Word": "Microsoft Word",
  "Microsoft Office": "Microsoft Office",
  "Apple Preview": "Preview",
  Preview: "Preview",
  Canva: "Canva",
  Google: "Google Drive",
  LibreOffice: "LibreOffice",
  Foxit: "Foxit",
  TinyWow: "TinyWow",
  Wondershare: "Wondershare",
  Nitro: "Nitro",
  Sejda: "Sejda",
};

function VerifyPageContent() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showMetadata, setShowMetadata] = useState(false);

  const [certLoading, setCertLoading] = useState(false);
  const [certResult, setCertResult] = useState(null);

  const detectEditor = (metadata) => {
    if (!metadata) return null;

    const fields = [
      metadata.creator,
      metadata.producer,
      metadata.application,
      metadata.CreatorTool,
      metadata.Producer,
      metadata.Creator,
    ].filter(Boolean);

    for (const field of fields) {
      if (typeof field === "string") {
        for (const [pattern, editor] of Object.entries(KNOWN_EDITORS)) {
          if (field.toLowerCase().includes(pattern.toLowerCase())) {
            return editor;
          }
        }
      }
    }
    return null;
  };

  const verifyCertificate = async (selectedFile) => {
    setCertLoading(true);
    setCertResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/ml/verify-certificate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => null);
        throw new Error(
          errBody?.detail || "Failed to verify certificate."
        );
      }

      const data = await response.json();
      setCertResult(data);
    } catch (err) {
      console.error("Certificate verification error:", err);
      setCertResult({
        valid: false,
        has_certificate: false,
        message: "Could not reach the certificate verification service. Please try again later.",
      });
    }

    setCertLoading(false);
  };

  const handleFileAnalysis = async (selectedFile) => {
    setLoading(true);
    setResults(null);
    setError(null);
    setCertResult(null);
    setShowMetadata(false);

    try {
      const verificationResult = await verifyDocument(selectedFile);
      const metadata = {
        ...(verificationResult.core || {}),
        modificationDate: verificationResult.core?.modDate || null,
      };
      delete metadata.modDate;

      const knownEditor =
        verificationResult.knownEditor || detectEditor(metadata);

      setResults({
        sha256: verificationResult.sha256,
        metadata: metadata || {},
        knownEditor,
        fileInfo: {
          name: selectedFile.name,
          size: selectedFile.size,
        },
      });
    } catch (err) {
      console.error("Error processing PDF:", err);
      setError(
        err.message || "Failed to process the PDF file. Please try again."
      );
    }
    setLoading(false);

    verifyCertificate(selectedFile);
  };

  const handleFileSelect = (selectedFile) => {
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
    handleFileAnalysis(selectedFile);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatConfidence = (score) => {
    if (score == null) return "N/A";
    return `${(score * 100).toFixed(1)}%`;
  };

  const formatDate = (isoString) => {
    if (!isoString) return "N/A";
    try {
      return new Date(isoString).toLocaleString();
    } catch {
      return isoString;
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResults(null);
    setError(null);
    setCertResult(null);
    setShowMetadata(false);
  };



return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
              <ShieldCheck className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-base-color">
              Verify & Inspect Document
            </h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a PDF to check if it holds a valid Kwiddex certificate and inspect its metadata.
          </p>
        </div>

        <div className="space-y-6">
          {!file && (
            <Card className="border-2 border-dashed border-border hover:border-blue-300 transition-colors">
              <CardContent className="p-0">
                <FileDropZone onFileSelect={handleFileSelect} loading={loading} />
              </CardContent>
            </Card>
          )}

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

          {file && (
            <>
              {/* Uploaded file info */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-600" /> Uploaded File
                    </CardTitle>
                    <Button variant="outline" size="sm" onClick={resetUpload}>
                      Upload New File
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <p className="font-mono text-sm bg-surface p-2 rounded break-all">
                      {file.name}
                    </p>
                    <p className="font-mono text-sm bg-surface p-2 rounded">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Certificate verification */}
              <AnimatePresence>
                {certLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card>
                      <CardContent className="py-8 text-center">
                        <Loader2 className="h-6 w-6 mx-auto animate-spin text-purple-600" />
                        <p className="mt-3 text-muted-foreground">
                          Checking for a Kwiddex certificate...
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {certResult && !certResult.has_certificate && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card className="border-border">
                      <CardContent className="py-6">
                        <div className="flex items-start gap-3">
                          <Info className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
                          <div>
                            <p className="font-medium text-base-color">
                              This document does not contain a Kwiddex certificate.
                            </p>
                            <p className="text-sm text-muted-foreground mt-1">
                              A certificate was not issued for this document using Kwiddex. If you expected one, make sure you're uploading the certified version of the file.
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}

                {certResult && certResult.has_certificate && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="space-y-4"
                  >
                    {/* Plain-language verdict */}
                    <Card
                      className={
                        certResult.valid
                          ? "border-green-200 bg-green-50 dark:bg-green-950/20 dark:border-green-800"
                          : "border-red-200 bg-red-50 dark:bg-red-950/20 dark:border-red-800"
                      }
                    >
                      <CardContent className="py-6">
                        <div className="flex items-start gap-3">
                          {certResult.valid ? (
                            <ShieldCheck className="w-6 h-6 text-green-600 dark:text-green-400 mt-0.5 shrink-0" />
                          ) : (
                            <ShieldAlert className="w-6 h-6 text-red-600 dark:text-red-400 mt-0.5 shrink-0" />
                          )}
                          <div>
                            <p
                              className={`text-lg font-semibold ${
                                certResult.valid
                                  ? "text-green-800 dark:text-green-300"
                                  : "text-red-800 dark:text-red-300"
                              }`}
                            >
                              {certResult.valid
                                ? "This document is certified by Kwiddex and has not been modified."
                                : certResult.signature_valid === false
                                ? "This certificate's signature is invalid. The document may have been tampered with."
                                : certResult.document_intact === false
                                ? "This document has been modified after certification."
                                : certResult.document_intact === false
                                ? "This document has been modified after certification."
                                : certResult.certificate_active === false
                                ? "This document's certificate has been revoked."
                                : "This certificate could not be verified."}
                            </p>
                            {certResult.valid && certResult.issued_at && (
                              <p className="text-sm text-green-700 dark:text-green-400 mt-1">
                                Certified on {formatDate(certResult.issued_at)}.
                              </p>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Certificate details */}
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="flex items-center gap-2 text-base">
                          <Fingerprint className="w-5 h-5 text-purple-600" />
                          Certificate Details
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          {certResult.certificate_id && (
                            <div className="flex items-start gap-2">
                              <Fingerprint className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                              <div>
                                <p className="text-xs text-muted-foreground">Certificate ID</p>
                                <p className="text-sm font-mono font-medium">{certResult.certificate_id}</p>
                              </div>
                            </div>
                          )}

                          {certResult.issued_at && (
                            <div className="flex items-start gap-2">
                              <Calendar className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                              <div>
                                <p className="text-xs text-muted-foreground">Issued</p>
                                <p className="text-sm font-medium">{formatDate(certResult.issued_at)}</p>
                              </div>
                            </div>
                          )}

                          {certResult.confidence_score != null && (
                            <div className="flex items-start gap-2">
                              <BarChart3 className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                              <div>
                                <p className="text-xs text-muted-foreground">CNN Confidence</p>
                                <p className="text-sm font-medium">{formatConfidence(certResult.confidence_score)}</p>
                              </div>
                            </div>
                          )}

                          {certResult.reviewer_id && (
                            <div className="flex items-start gap-2">
                              <User className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                              <div>
                                <p className="text-xs text-muted-foreground">Certified By</p>
                                <p className="text-sm font-medium">{certResult.reviewer_id}</p>
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-border">
                          {certResult.signature_valid != null && (
                            <Badge
                              variant="secondary"
                              className={
                                certResult.signature_valid
                                  ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                                  : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                              }
                            >
                              Signature {certResult.signature_valid ? "Valid" : "Invalid"}
                            </Badge>
                          )}
                          {certResult.document_intact != null && (
                            <Badge
                              variant="secondary"
                              className={
                                certResult.document_intact
                                  ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                                  : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                              }
                            >
                              Document {certResult.document_intact ? "Intact" : "Modified"}
                            </Badge>
                          )}
                          {certResult.certificate_active != null && (
                            <Badge
                              variant="secondary"
                              className={
                                certResult.certificate_active
                                  ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                                  : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
                              }
                            >
                              {certResult.certificate_active ? "Active" : "Revoked"}
                            </Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Metadata results */}
              <AnimatePresence>
                {loading && !results && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Card>
                      <CardContent className="py-12 text-center">
                        <Loader2 className="h-8 w-8 mx-auto animate-spin text-blue-600" />
                        <p className="mt-4 text-muted-foreground">
                          Analyzing document...
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
                {results && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="space-y-6"
                  >
                    {results.knownEditor && (
                      <Card>
                        <CardHeader className="pb-3">
                          <CardTitle className="flex items-center gap-2">
                            <Wrench className="w-5 h-5 text-orange-600" /> Editor Detection
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                            {results.knownEditor} Detected
                          </Badge>
                        </CardContent>
                      </Card>
                    )}

                    {!showMetadata ? (
                      <Button
                        onClick={() => setShowMetadata(true)}
                        variant="outline"
                        className="w-full"
                      >
                        <FileText className="w-4 h-4 mr-2" />
                        View Full Metadata
                      </Button>
                    ) : (
                      <div className="space-y-4">
                        <Card>
                          <CardHeader className="pb-3">
                            <CardTitle className="flex items-center gap-2">
                              <Hash className="w-5 h-5 text-green-600" /> File Integrity Hash
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="font-mono text-sm bg-surface p-3 rounded break-all">
                              {results.sha256}
                            </div>
                          </CardContent>
                        </Card>
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



