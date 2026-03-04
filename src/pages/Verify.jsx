
import React, { useState } from "react";
import { verifyDocument } from "@/api/verify";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { 
  Upload, 
  FileText, 
  Shield, 
  Copy, 
  CheckCircle, 
  AlertCircle,
  Hash,
  Wrench,
  Fingerprint,
  ShieldCheck,
  ShieldAlert,
  User,
  Loader2,
  Newspaper
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
  "Preview": "Preview",
  "Canva": "Canva",
  "Google": "Google Drive",
  "LibreOffice": "LibreOffice",
  "Foxit": "Foxit",
  "TinyWow": "TinyWow",
  "Wondershare": "Wondershare",
  "Nitro": "Nitro",
  "Sejda": "Sejda"
};

function VerifyPageContent() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copySuccess, setCopySuccess] = useState("");
  const [showMetadata, setShowMetadata] = useState(false);
  
  const [verificationCode, setVerificationCode] = useState("");
  const [verificationResult, setVerificationResult] = useState(null);

  const detectEditor = (metadata) => {
    if (!metadata) return null;
    
    const fields = [
      metadata.creator,
      metadata.producer, 
      metadata.application,
      metadata.CreatorTool,
      metadata.Producer,
      metadata.Creator
    ].filter(Boolean);

    for (const field of fields) {
      if (typeof field === 'string') {
        for (const [pattern, editor] of Object.entries(KNOWN_EDITORS)) {
          if (field.toLowerCase().includes(pattern.toLowerCase())) {
            return editor;
          }
        }
      }
    }
    return null;
  };

  const computeFileHash = async (file) => {
    const arrayBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  };

  const handleFileAnalysis = async (selectedFile) => {
    setLoading(true);
    setResults(null);
    setError(null);
    setVerificationResult(null);
    setShowMetadata(false);

    try {
      const verificationResult = await verifyDocument(selectedFile);
      const metadata = {
        ...(verificationResult.core || {}),
        modificationDate: verificationResult.core?.modDate || null,
      };
      delete metadata.modDate;

      const knownEditor = verificationResult.knownEditor || detectEditor(metadata);

      setResults({
        sha256: verificationResult.sha256,
        metadata: metadata || {},
        knownEditor,
        fileInfo: {
          name: selectedFile.name,
          size: selectedFile.size,
        }
      });

    } catch (err) {
      console.error("Error processing PDF:", err);
      setError(err.message || "Failed to process the PDF file. Please try again.");
    }
    setLoading(false);
  };
  
  const handleFileSelect = (selectedFile) => {
      if (!selectedFile) return;
      if (selectedFile.type !== 'application/pdf') {
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

  const handleVerification = async () => {
    if (!file || !verificationCode) {
        setError("Please upload a file and provide a signature code to verify.");
        return;
    }
    setError(null);
    setVerificationResult(null);

    try {
        const currentHash = await computeFileHash(file);
        const decodedString = atob(verificationCode);
        const [originalHash, userEmail, timestamp] = decodedString.split('@');

        if (currentHash === originalHash) {
            setVerificationResult({
                status: 'success',
                message: 'Document is authentic and has not been modified.',
                signedBy: userEmail,
                signedOn: new Date(timestamp).toLocaleString()
            });
        } else {
            setVerificationResult({
                status: 'failed',
                message: 'Verification failed. The document has been modified or is not the correct file.',
                signedBy: userEmail,
                signedOn: new Date(timestamp).toLocaleString()
            });
        }

    } catch (e) {
        setVerificationResult({
            status: 'error',
            message: 'Invalid signature code. Please check the code and try again.'
        });
        console.error("Verification error:", e);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const resetUpload = () => {
    setFile(null);
    setResults(null);
    setError(null);
    setVerificationCode("");
    setVerificationResult(null);
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
                Upload a PDF to inspect its metadata and verify its authenticity using a signature code.
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
              <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {file && (
            <>
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="w-5 h-5 text-blue-600" /> Uploaded File
                      </CardTitle>
                      <Button variant="outline" size="sm" onClick={resetUpload}> Upload New File </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between">
                        <p className="font-mono text-sm bg-surface p-2 rounded break-all">{file.name}</p>
                        <p className="font-mono text-sm bg-surface p-2 rounded">{formatFileSize(file.size)}</p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2">
                        <Fingerprint className="w-5 h-5 text-purple-600" /> Verify Document Signature
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                        <p className="text-sm text-muted-foreground">Paste the signature code to verify the document's authenticity.</p>
                        <div>
                            <label htmlFor="verification-code" className="text-sm font-medium text-muted-foreground">Signature Code</label>
                            <Input id="verification-code" value={verificationCode} onChange={(e) => setVerificationCode(e.target.value)} placeholder="Paste signature code here..." className="mt-1" />
                        </div>
                        <Button onClick={handleVerification} disabled={!verificationCode}>
                            <ShieldCheck className="w-4 h-4 mr-2"/> Verify Authenticity
                        </Button>
                    </div>
                  </CardContent>
                </Card>

                <AnimatePresence>
                {verificationResult && (
                    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                        <Card className={`${ verificationResult.status === 'success' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200' }`}>
                            <CardHeader>
                                <CardTitle className={`flex items-center gap-2 ${ verificationResult.status === 'success' ? 'text-green-700' : 'text-red-700' }`}>
                                    {verificationResult.status === 'success' ? <ShieldCheck /> : <ShieldAlert />}
                                    Verification Result
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="font-semibold text-base-color">{verificationResult.message}</p>
                                {verificationResult.signedBy && (
                                  <p className="text-sm mt-2 text-base-color">
                                    Digitally signed by <span className="font-medium">{verificationResult.signedBy}</span> on <span className="font-medium">{verificationResult.signedOn}</span>.
                                  </p>
                                )}
                            </CardContent>
                        </Card>
                    </motion.div>
                )}
                </AnimatePresence>

                <AnimatePresence>
                    {loading && !results && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <Card><CardContent className="py-12 text-center">
                                <Loader2 className="h-8 w-8 mx-auto animate-spin text-blue-600" />
                                <p className="mt-4 text-muted-foreground">Analyzing document... (this may take a moment)</p>
                            </CardContent></Card>
                        </motion.div>
                    )}
                    {results && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                        <Card>
                          <CardHeader className="pb-3"><CardTitle className="flex items-center gap-2"><Hash className="w-5 h-5 text-green-600" /> File Integrity Hash</CardTitle></CardHeader>
                          <CardContent>
                                <div className="font-mono text-sm bg-surface p-3 rounded flex-1 break-all">{results.sha256}</div>
                          </CardContent>
                        </Card>
                        {results.knownEditor && (
                          <Card>
                            <CardHeader className="pb-3"><CardTitle className="flex items-center gap-2"><Wrench className="w-5 h-5 text-orange-600" /> Editor Detection</CardTitle></CardHeader>
                            <CardContent><Badge variant="secondary" className="bg-orange-100 text-orange-800">{results.knownEditor} Detected</Badge></CardContent>
                          </Card>
                        )}
                        
                        {!showMetadata ? (
                             <Button onClick={() => setShowMetadata(true)} variant="outline" className="w-full">
                                <FileText className="w-4 h-4 mr-2" />
                                View Full Metadata
                            </Button>
                        ) : (
                            <MetadataTable metadata={results.metadata} />
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
