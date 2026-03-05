import React, { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, Loader2 } from "lucide-react";

export default function FileDropZone({ onFileSelect, loading }) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      const file = files[0];
      if (file.type === "application/pdf") {
        onFileSelect(file);
      }
    }
  };

  const handleFileInput = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={`relative transition-all duration-200 ${
        dragActive ? "bg-blue-50 border-blue-300" : "hover:bg-surface"
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        onChange={handleFileInput}
        className="hidden"
        disabled={loading}
      />
      
      <div className="p-8 md:p-12 text-center">
        <div className="max-w-md mx-auto">
          {loading ? (
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-base-color">Processing...</h3>
                <p className="text-sm text-gray-600">Please wait while we analyze your PDF</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center transition-colors ${
                dragActive ? "bg-blue-100" : "bg-surface"
              }`}>
                <FileText className={`w-8 h-8 transition-colors ${
                  dragActive ? "text-blue-600" : "text-gray-400"
                }`} />
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-base-color mb-2">
                  {dragActive ? "Drop your PDF here" : "Upload PDF for Verification"}
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Drag and drop your PDF file here, or click to browse
                </p>
                
                <Button
                  onClick={openFileDialog}
                  className="bg-blue-600 hover:bg-blue-700"
                  disabled={loading}
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Choose PDF File
                </Button>
              </div>
              
              <div className="text-xs text-muted-foreground space-y-1">
                <p>• Supports PDF files up to 25MB</p>
                <p>• Secure processing - files are not stored permanently</p>
                <p>• Metadata extraction and integrity verification</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}