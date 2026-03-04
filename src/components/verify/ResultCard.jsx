import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Copy, CheckCircle, Hash, FileText } from "lucide-react";

export default function ResultCard({ results, onCopy, copySuccess }) {
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* File Information */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            File Information
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-500">Filename</label>
              <p className="font-mono text-sm bg-gray-50 p-2 rounded mt-1 break-all">
                {results.fileInfo.name}
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Size</label>
              <p className="font-mono text-sm bg-gray-50 p-2 rounded mt-1">
                {formatFileSize(results.fileInfo.size)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* SHA-256 Hash */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Hash className="w-5 h-5 text-green-600" />
            File Integrity Hash
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium text-gray-500">SHA-256 (hex)</label>
              <div className="flex items-center gap-2 mt-1">
                <div className="font-mono text-sm bg-gray-50 p-3 rounded flex-1 break-all">
                  {results.sha256}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onCopy(results.sha256, "SHA-256")}
                  className="flex items-center gap-1"
                >
                  {copySuccess === "SHA-256" ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy
                    </>
                  )}
                </Button>
              </div>
            </div>
            <p className="text-sm text-gray-600">
              This hash can be used to verify file integrity and detect any modifications.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Known Editor Detection */}
      {results.knownEditor && (
        <Card>
          <CardHeader>
            <CardTitle>Editor Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-3">
              <Badge variant="secondary" className="bg-orange-100 text-orange-800">
                {results.knownEditor} Detected
              </Badge>
              <span className="text-sm text-gray-600">
                This PDF appears to have been created or edited with {results.knownEditor}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-2">
              * Detection is heuristic based on metadata and not definitive proof of editing
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}