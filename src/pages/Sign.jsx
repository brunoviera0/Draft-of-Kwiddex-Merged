import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Fingerprint, Copy, CheckCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import FileDropZone from "../components/verify/FileDropZone";
import AuthGuard from "../components/auth/AuthGuard";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/use-toast";
import { API_BASE } from "@/api/verify";

function SignPageContent() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [signatureCode, setSignatureCode] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState(null);
  const [email, setEmail] = useState("");

  const computeFileHash = async (fileToHash) => {
    const arrayBuffer = await fileToHash.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  };

  const sendSignatureEmail = async ({
    to,
    signerEmail,
    signatureCode: generatedCode,
    documentId,
    documentUrl,
  }) => {
    const endpointBase = API_BASE || '';
    const response = await fetch(`${endpointBase}/api/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ to, signerEmail, signatureCode: generatedCode, documentId, documentUrl }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.error || 'Failed to send email');
    }
    return data;
  };

  const handleFileSelect = async (selectedFile) => {
    if (!selectedFile) return;

    if (selectedFile.type !== 'application/pdf') {
      setError("Please select a PDF file only.");
      return;
    }
    if (selectedFile.size > 25 * 1024 * 1024) {
      setError("File size must be less than 25MB.");
      return;
    }

    if (!email.trim()) {
      setError("Please enter your email address before generating a signature code.");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setSignatureCode(null);
    setLoading(true);

    try {
      const sha256Hash = await computeFileHash(selectedFile);
      const timestamp = new Date().toISOString();
      const userEmail = email.trim();

      const rawSignature = `${sha256Hash}@${userEmail}@${timestamp}`;
      const encodedSignature = btoa(rawSignature);

      setSignatureCode(encodedSignature);

      try {
        setSending(true);
        await sendSignatureEmail({
          to: userEmail,
          signerEmail: userEmail,
          signatureCode: encodedSignature,
          documentId: null,
          documentUrl: null,
        });
        toast({
          title: "Email sent!",
          description: "We just sent your signature code to your inbox.",
        });
      } catch (emailError) {
        console.error("Failed to send signature email:", emailError);
        toast({
          title: "Could not send email",
          description: emailError.message || "Please try again.",
          variant: "destructive",
        });
      } finally {
        setSending(false);
      }
    } catch (err) {
      console.error("Error generating signature:", err);
      setError("Failed to generate the signature code. Please try again.");
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
      setSignatureCode(null);
      setError(null);
      setLoading(false);
  }

  return (
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-3 mb-4">
                <div className="w-12 h-12 bg-green-600 rounded-xl flex items-center justify-center">
                    <Fingerprint className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-3xl md:text-4xl font-bold text-base-color">
                    Sign & Secure Document
                </h1>
            </div>
            <p className="text-muted-foreground max-w-2xl mx-auto">
                Upload a PDF to generate a unique, verifiable "signature" code. This code acts as a digital signature to prove the document's authenticity and integrity.
            </p>
        </div>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="text-left text-lg font-semibold">Signer Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <label className="text-sm font-medium text-muted-foreground" htmlFor="signer-email">Email Address</label>
            <Input
              id="signer-email"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              disabled={loading || sending}
            />
            <p className="text-xs text-muted-foreground">Your email is embedded in the signature code so recipients can confirm who generated it.</p>
          </CardContent>
        </Card>

        <AnimatePresence>
          {error && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="mb-4">
              <div className="flex items-center gap-2 bg-red-50 text-red-700 px-4 py-3 rounded-lg">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                </svg>
                <span className="text-sm font-medium">{error}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {!signatureCode ? (
            <Card className="border-2 border-dashed border-border hover:border-green-300 transition-colors">
              <CardContent className="p-0">
                <FileDropZone
                  onFileSelect={handleFileSelect}
                  loading={loading || sending}
                />
              </CardContent>
            </Card>
        ) : (
            <AnimatePresence>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-6"
                >
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <CheckCircle className="w-6 h-6 text-green-600" />
                                Signature Code Generated
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-muted-foreground mb-4">This code can be used to verify the integrity and origin of <span className="font-semibold text-base-color">{file.name}</span>.</p>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-muted-foreground">Your Document Signature Code</label>
                                <div className="font-mono text-sm bg-surface p-4 rounded-lg break-all">
                                    {signatureCode}
                                </div>
                            </div>

                            <div className="flex items-center gap-4 mt-6">
                                <Button onClick={() => copyToClipboard(signatureCode)}>
                                    {copySuccess ? (
                                        <>
                                            <CheckCircle className="w-4 h-4 mr-2" /> Copied
                                        </>
                                    ) : (
                                        <>
                                            <Copy className="w-4 h-4 mr-2" /> Copy Code
                                        </>
                                    )}
                                </Button>
                                <Button variant="outline" onClick={reset} disabled={loading || sending}>
                                    Sign Another Document
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                </motion.div>
            </AnimatePresence>
        )}
      </div>
    </div>
  );
}

export default function SignPage() {
    return <AuthGuard><SignPageContent /></AuthGuard>;
}
