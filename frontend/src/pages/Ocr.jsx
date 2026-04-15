import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Newspaper, Copy, CheckCircle, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import FileDropZone from "../components/verify/FileDropZone";

import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfjsWorker;

function OcrPageContent() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [extractedText, setExtractedText] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = async (selectedFile) => {
    if (!selectedFile) return;
    if (selectedFile.type !== 'application/pdf') {
      setError("Please select a PDF file only.");
      return;
    }
    if (selectedFile.size > 50 * 1024 * 1024) {
      setError("File size must be less than 50MB.");
      return;
    }

    setFile(selectedFile);
    setError(null);
    setExtractedText(null);
    setLoading(true);

    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const pdf = await getDocument({ data: new Uint8Array(arrayBuffer) }).promise;
      let textContent = "";

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        const pageText = content.items
          .map((item) => ("str" in item ? item.str : ""))
          .join(" ")
          .trim();

        if (pageText) {
          textContent += `${pageText}\n\n`;
        }
      }

      if (textContent.trim().length === 0) {
        setError("No text content could be extracted from this PDF.");
        setExtractedText("No text content could be extracted.");
      } else {
        setExtractedText(textContent.trim());
      }
    } catch (err) {
      console.error("Error during OCR:", err);
      setError("Failed to process the document for OCR. Please try again.");
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
    <div className="min-h-screen bg-base text-base-color p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-3 mb-4">
                <div className="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center">
                    <Newspaper className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-3xl md:text-4xl font-bold text-base-color">
                    Extract Text (OCR)
                </h1>
            </div>
            <p className="text-muted-foreground max-w-2xl mx-auto">
                Upload a PDF to extract its text content using Optical Character Recognition (OCR). Ideal for scanned documents.
            </p>
        </div>

        <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="mb-4">
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              </motion.div>
            )}
        </AnimatePresence>
        
        {extractedText === null ? (
          <Card className="border-2 border-dashed border-border hover:border-indigo-300 transition-colors">
            <CardContent className="p-0">
              <FileDropZone
                onFileSelect={handleFileSelect}
                loading={loading}
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
                    <CheckCircle className="w-6 h-6 text-indigo-600" />
                    Text Extracted Successfully
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4">Full text content from <span className="font-semibold text-base-color">{file.name}</span>:</p>

                  <div className="relative">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copyToClipboard(extractedText)}
                      className="absolute top-2 right-2 z-10"
                    >
                      {copySuccess ? (
                        <><CheckCircle className="w-4 h-4 mr-2 text-green-500" /> Copied</>
                      ) : (
                        <><Copy className="w-4 h-4 mr-2" /> Copy All Text</>
                      )}
                    </Button>
                    <pre className="w-full h-96 overflow-y-auto bg-surface p-4 rounded-md whitespace-pre-wrap font-sans text-sm text-base-color">
                      {extractedText}
                    </pre>
                  </div>

                  <div className="mt-6">
                    <Button variant="outline" onClick={reset}>
                        Extract from Another Document
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

export default function OcrPage() {
    return <OcrPageContent />;
}
