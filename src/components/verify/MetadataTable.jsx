
import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, User, FileText, Tag } from "lucide-react";

export default function MetadataTable({ metadata }) {
  const formatDate = (dateString) => {
    if (!dateString) return "Not available";
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString; // Return as-is if can't parse
    }
  };

  const metadataFields = [
    { key: "title", label: "Title", icon: FileText },
    { key: "author", label: "Author", icon: User },
    { key: "creator", label: "Creator", icon: User },
    // 'producer' and 'CreatorTool' typically refer to software, not a person.
    // Changing their icons from User to FileText for better semantic representation.
    { key: "producer", label: "Producer", icon: FileText },
    { key: "CreatorTool", label: "Creator Tool", icon: FileText },
    // 'Producer' (capitalized) is likely a variant of 'producer', representing software.
    { key: "Producer", label: "Producer", icon: FileText },
    // 'Creator' (capitalized) if referring to a person, should use User icon.
    { key: "Creator", label: "Creator", icon: User },
    { key: "subject", label: "Subject", icon: Tag },
    { key: "keywords", label: "Keywords", icon: Tag },
    { key: "application", label: "Application", icon: FileText },
    { key: "creationDate", label: "Creation Date", icon: Calendar, isDate: true },
    { key: "modificationDate", label: "Modification Date", icon: Calendar, isDate: true }
  ];

  const hasMetadata = Object.values(metadata).some(value => 
    value !== null && value !== undefined && value !== ""
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          PDF Metadata
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!hasMetadata ? (
          <div className="text-center py-8">
            <div className="w-16 h-16 mx-auto mb-4 bg-surface rounded-full flex items-center justify-center">
              <FileText className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500">No metadata found in this PDF</p>
            <p className="text-sm text-gray-400 mt-1">
              This PDF may not contain standard metadata fields
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {metadataFields.map(({ key, label, icon: Icon, isDate }) => {
              const value = metadata[key];
              if (!value) return null;

              return (
                <div key={key} className="flex items-start gap-3 p-3 bg-surface rounded-lg">
                  <Icon className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <label className="text-sm font-medium text-gray-700">
                      {label}
                    </label>
                    <p className="text-sm text-base-color mt-1 break-words">
                      {isDate ? formatDate(value) : value}
                    </p>
                  </div>
                </div>
              );
            })}

            {/* Show additional unknown fields */}
            {Object.entries(metadata).map(([key, value]) => {
              // Ensure that value is not empty and key is not already handled by predefined fields
              if (!value || metadataFields.some(field => field.key === key)) return null;

              return (
                <div key={key} className="flex items-start gap-3 p-3 bg-surface rounded-lg">
                  <Tag className="w-4 h-4 text-gray-500 mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <label className="text-sm font-medium text-gray-700 capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </label>
                    <p className="text-sm text-base-color mt-1 break-words">
                      {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
