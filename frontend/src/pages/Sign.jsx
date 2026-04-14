import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Fingerprint,
  CheckCircle,
  Download,
  Loader2,
  AlertCircle,
  Calendar,
  BarChart3,
  User,
  FileText,
  ShieldCheck,
  Mail,
  ScanLine,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import FileDropZone from "../components/verify/FileDropZone";

import { toast } from "@/components/ui/use-toast";
import { API_BASE } from "@/api/verify";
import { useAuth } from "@/context/AuthContext";

export default function SignPage() {
  const { user, getToken, isAuthenticated, login } = useAuth();
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);

  // Step 1: Analysis state
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Step 2: Certification state
  const [certifying, setCertifying] = useState(false);
  const [certifiedBlobUrl, setCertifiedBlobUrl] = useState(null);
  const [certifiedFilename, setCertifiedFilename] = useState(null);
  const [certificateDetails, setCertificateDetails] = useState(null);

  // Email state
  const [sendingEmail, setSendingEmail] = useState(false);

  // Step 1: Analyze the document
  const handleFileSelect = async (selectedFile) => {
    if (!isAuthenticated) {
      login();
      return;
    }
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
    setAnalysisResult(null);
    setCertifiedBlobUrl(null);
    setCertificateDetails(null);
    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE}/api/physical/score`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = "Analysis failed. Please try again.";
        try {
          const errBody = await response.json();
          if (errBody?.detail) errorMsg = errBody.detail;
        } catch {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setAnalysisResult(data);
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err.message || "Failed to analyze the document.");
    }

    setAnalyzing(false);
  };

  // Step 2: Certify after user reviews
  const handleCertify = async () => {
    if (!file) return;

    setCertifying(true);
    setError(null);

    try {
      const token = await getToken();
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/ml/certify`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = "Certification failed. Please try again.";
        try {
          const errBody = await response.json();
          if (errBody?.detail) errorMsg = errBody.detail;
        } catch {}
        throw new Error(errorMsg);
      }

      const certificateId = response.headers.get("X-Certificate-ID");
      const confidenceScore = response.headers.get("X-Confidence-Score");

      const disposition = response.headers.get("Content-Disposition");
      let filename = `${file.name.replace(/\.pdf$/i, "")}_certified.pdf`;
      if (disposition) {
        const match = disposition.match(/filename="?([^"]+)"?/);
        if (match) filename = match[1];
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);

      setCertifiedBlobUrl(blobUrl);
      setCertifiedFilename(filename);
      setCertificateDetails({
        certificate_id: certificateId,
        confidence_score: confidenceScore ? parseFloat(confidenceScore) : null,
        issued_at: new Date().toISOString(),
        reviewer_id: user?.email || user?.id || "Unknown",
      });
    } catch (err) {
      console.error("Certification error:", err);
      setError(err.message || "Failed to certify the document.");
    }

    setCertifying(false);
  };

  const handleDownload = () => {
    if (!certifiedBlobUrl) return;
    const a = document.createElement("a");
    a.href = certifiedBlobUrl;
    a.download = certifiedFilename || "certified.pdf";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleSendEmail = async () => {
    if (!certificateDetails || !user?.email) return;

    setSendingEmail(true);
    try {
      const token = await getToken();
      const response = await fetch(`${API_BASE}/api/email`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          to: user.email,
          signerEmail: user.email,
          signatureCode: certificateDetails.certificate_id,
          documentId: null,
          documentUrl: null,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.error || "Failed to send email");
      }

      toast({
        title: "Email sent!",
        description: "Certificate details have been sent to your inbox.",
      });
    } catch (err) {
      console.error("Email error:", err);
      toast({
        title: "Could not send email",
        description: err.message || "Please try again.",
        variant: "destructive",
      });
    }
    setSendingEmail(false);
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

  const reset = () => {
    if (certifiedBlobUrl) URL.revokeObjectURL(certifiedBlobUrl);
    setFile(null);
    setAnalysisResult(null);
    setCertifiedBlobUrl(null);
    setCertifiedFilename(null);
    setCertificateDetails(null);
    setError(null);
    setAnalyzing(false);
    setCertifying(false);
  };

  // Determine current step
  const showUpload = !file || (!analysisResult && !analyzing);
  const showAnalysis = analysisResult && !certificateDetails;
  const showCertified = !!certificateDetails;

  return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
              <Fingerprint className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-base-color">
              Certify Document
            </h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload a PDF to run it through our CNN analysis. Review the authenticity results, then choose to certify the document with a Kwiddex-signed certificate.
          </p>
          {user?.email && (
            <p className="mt-2 text-sm text-muted-foreground">
              Certifying as{" "}
              <span className="font-medium text-foreground">{user.email}</span>
            </p>
          )}
        </div>


<AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-4"
            >
              <div className="flex items-center gap-2 bg-red-50 dark:bg-red-950/20 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span className="text-sm font-medium">{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 1: Upload */}
        {showUpload && (
          <Card className="border-2 border-dashed border-border hover:border-green-300 transition-colors">
            <CardContent className="p-0">
      
      {!isAuthenticated && (
        <Card className="border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/30 mb-6">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Sign in required to certify documents
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                You can browse this page to learn about the certification process. To certify a document, please sign in first.
              </p>
            </div>
            <Button
              size="sm"
              className="bg-blue-600 hover:bg-blue-700 text-white shrink-0"
              onClick={() => login()}
            >
              Sign In
            </Button>
          </CardContent>
        </Card>
      )}
        <FileDropZone onFileSelect={handleFileSelect} loading={analyzing} title="Upload PDF for Certification" description="Drag and drop your PDF here to analyze and certify" acceptLabel="Choose PDF File" />
              {analyzing && (
                <div className="px-6 pb-6 text-center">
                  <p className="text-sm text-muted-foreground">
                    Running CNN analysis on your document...
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Step 2: Analysis results — user reviews before certifying */}
        {showAnalysis && (
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* File info */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="w-5 h-5 text-blue-600" /> Uploaded File
                    </CardTitle>
                    <Button variant="outline" size="sm" className="border-border text-foreground" onClick={reset}>
                      Upload Different File
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="font-mono text-sm bg-surface p-2 rounded break-all">
                    {file.name}
                  </p>
                </CardContent>
              </Card>

              {/* Analysis results */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2">
                    <ScanLine className="w-5 h-5 text-blue-600" /> Analysis Results
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <p className="text-xs text-muted-foreground uppercase tracking-wide">Confidence</p>
                      <p className="text-2xl font-bold mt-1">
                        {formatConfidence(analysisResult.confidence)}
                      </p>
                    </div>

                    {analysisResult.confidenceInterval && (
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">95% Confidence Interval</p>
                        <p className="text-lg font-semibold mt-1">
                          {formatConfidence(analysisResult.confidenceInterval.lower)} — {formatConfidence(analysisResult.confidenceInterval.upper)}
                        </p>
                      </div>
                    )}

                    {analysisResult.monteCarloStats && (
                      <div>
                        <p className="text-xs text-muted-foreground uppercase tracking-wide">Monte Carlo Stats</p>
                        <div className="mt-1 space-y-1 text-sm font-medium">
                          <p>Samples: {analysisResult.monteCarloStats.numSamples}</p>
                          <p>Agreement: {formatConfidence(analysisResult.monteCarloStats.agreementRate)}</p>
                          <p>Std Dev: {analysisResult.monteCarloStats.stdDev.toFixed(4)}</p>
                        </div>
                      </div>
                    )}

                    <p className="text-xs text-muted-foreground">Model: {analysisResult.model} | Provider: {analysisResult.provider}</p>

                    <div className="pt-4 border-t border-border">
                      <p className="text-sm text-muted-foreground">
                        Review these results alongside you're own judgment of the document. If you are satisfied, you can certify this document below. Certifying attaches a Kwiddex-signed certificate to the PDF, which ties your account to this verification. Use with your own professional judgment. Kwiddex provides the analysis tool, but the decision to certify is yours.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Certify action */}
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  onClick={handleCertify}
                  disabled={certifying}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {certifying ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <ShieldCheck className="w-4 h-4 mr-2" />
                  )}
                  {certifying ? "Certifying..." : "Certify This Document"}
                </Button>
                <Button variant="outline" className="border-border text-foreground" onClick={reset} disabled={certifying}>
                  Cancel
                </Button>
              </div>
            </motion.div>
          </AnimatePresence>
        )}

        {/* Step 3: Certification complete */}
        {showCertified && (
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* Success banner */}
              <Card className="border-green-200 bg-green-50 dark:bg-green-950/20 dark:border-green-800">
                <CardContent className="py-6">
                  <div className="flex items-start gap-3">
                    <ShieldCheck className="w-6 h-6 text-green-600 dark:text-green-400 mt-0.5 shrink-0" />
                    <div>
                      <p className="text-lg font-semibold text-green-800 dark:text-green-300">
                        Document certified successfully.
                      </p>
                      <p className="text-sm text-green-700 dark:text-green-400 mt-1">
                        Your certified PDF contains an embedded Kwiddex certificate with a digital signature. Download it below.
                      </p>
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
                    {certificateDetails.certificate_id && (
                      <div className="flex items-start gap-2">
                        <Fingerprint className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                        <div>
                          <p className="text-xs text-muted-foreground">Certificate ID</p>
                          <p className="text-sm font-mono font-medium">
                            {certificateDetails.certificate_id}
                          </p>
                        </div>
                      </div>
                    )}

                    <div className="flex items-start gap-2">
                      <Calendar className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                      <div>
                        <p className="text-xs text-muted-foreground">Issued</p>
                        <p className="text-sm font-medium">
                          {formatDate(certificateDetails.issued_at)}
                        </p>
                      </div>
                    </div>

                    {certificateDetails.confidence_score != null && (
                      <div className="flex items-start gap-2">
                        <BarChart3 className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                        <div>
                          <p className="text-xs text-muted-foreground">CNN Confidence</p>
                          <p className="text-sm font-medium">
                            {formatConfidence(certificateDetails.confidence_score)}
                          </p>
                        </div>
                      </div>
                    )}

                    <div className="flex items-start gap-2">
                      <User className="w-4 h-4 mt-0.5 text-muted-foreground shrink-0" />
                      <div>
                        <p className="text-xs text-muted-foreground">Certified By</p>
                        <p className="text-sm font-medium">
                          {certificateDetails.reviewer_id}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-border">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <FileText className="w-4 h-4" />
                      <span>
                        Original file:{" "}
                        <span className="font-medium text-base-color">
                          {file?.name}
                        </span>
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Actions */}
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  onClick={handleDownload}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Certified PDF
                </Button>

                <div className="relative group">
                  <Button
                    variant="outline"
                    className="border-border text-muted-foreground cursor-not-allowed opacity-60"
                    disabled
                  >
                    <Mail className="w-4 h-4 mr-2" />
                    Email Certificate Details
                  </Button>
                  <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 text-xs bg-card border border-border text-muted-foreground rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                    Email notifications are not yet configured
                  </span>
                </div>

                <Button variant="outline" className="border-border text-foreground" onClick={reset}>
                  Certify Another Document
                </Button>
              </div>
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}


