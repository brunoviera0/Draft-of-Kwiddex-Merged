import React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Fingerprint,
  FileCheck,
  Scale,
  Shield,
  ArrowRight,
  Lock,
  Eye,
  Cpu,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.12, duration: 0.6, ease: "easeOut" },
  }),
};

const features = [
  {
    title: "Certify",
    path: "/sign",
    icon: Fingerprint,
    color: "from-blue-500/20 to-blue-600/5",
    accent: "text-blue-300",
    border: "border-blue-500/20 hover:border-blue-500/40",
    description:
      "After analyzing a document, certify it with a Kwiddex RSA-signed certificate. The certificate is embedded in the PDF metadata along with a visible certificate page. The certified file hash is stored for future integrity verification.",
    howTo:
      "Sign in, upload a PDF, review the CNN analysis results, then click Certify This Document. The certified PDF is returned for download with the embedded certificate.",
    details: ["RSA-2048 signatures", "SHA-256 hashing", "Tamper detection", "Login required"],
  },
  {
    title: "Verify",
    path: "/verify",
    icon: FileCheck,
    color: "from-emerald-500/15 to-emerald-500/5",
    accent: "text-emerald-600",
    border: "border-emerald-500/20 hover:border-emerald-500/40",
    description:
      "Upload any PDF to check if it contains a valid Kwiddex certificate. The system extracts the embedded certificate and RSA signature, verifies the signature, checks for revocation, and compares the file hash to detect post-certification modifications.",
    howTo:
      "Upload a PDF file. The system automatically checks for a Kwiddex certificate and displays the result with full metadata extraction.",
    details: ["RSA signature verification", "Integrity checking", "Revocation lookup", "PDF metadata extraction"],
  },
  {
    title: "Compare",
    path: "/compare",
    icon: Scale,
    color: "from-purple-500/20 to-purple-500/5",
    accent: "text-purple-500",
    border: "border-purple-500/20 hover:border-purple-500/40",
    description:
      "Upload two document images for side-by-side multi-region forensic comparison. The engine normalizes, aligns, and analyzes artwork, text, borders, and print texture to produce a detailed similarity report with verdict.",
    howTo:
      "Upload a reference and compared document image. Adjust alignment radius and brightness normalization. Click Compare to run the multi-region analysis.",
    details: ["Multi-region analysis", "Alignment + normalization", "Heatmap + micro regions", "Runs client-side"],
  },
];

const principles = [
  {
    icon: Lock,
    title: "Zero PII Storage",
    text: "Documents are never stored. Only metadata, hashes, and certification records are retained.",
  },
  {
    icon: Eye,
    title: "Full Transparency",
    text: "Certificate verification is public. Dispute history is permanently visible to all parties.",
  },
  {
    icon: Cpu,
    title: "Statistical Rigor",
    text: "Monte Carlo inference with confidence intervals. No single-pass binary predictions.",
  },
  {
    icon: Shield,
    title: "Tamper Detection",
    text: "SHA-256 hashing detects any modification to a certified document, down to a single byte.",
  },
];

export default function Home() {
  const { isAuthenticated, login } = useAuth();

  return (
    <div className="min-h-screen">
      <section className="relative overflow-hidden py-20 md:py-32">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-blue-500/10 rounded-full blur-[140px]" />
          <div className="absolute bottom-0 left-1/4 w-[400px] h-[400px] bg-blue-600/8 rounded-full blur-[120px]" />
        </div>

        <div className="relative max-w-5xl mx-auto px-6 text-center">
          <motion.h1
            className="text-4xl sm:text-5xl md:text-7xl font-bold tracking-tight text-base-color mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.6 }}
          >
            Kwiddex
          </motion.h1>

          <motion.p
            className="text-lg sm:text-xl md:text-2xl text-blue-400 font-medium mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            A tool to aid with forensic document analysis
          </motion.p>

          <motion.p
            className="text-muted-foreground max-w-2xl mx-auto text-sm sm:text-base md:text-lg leading-relaxed mb-10 px-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            Certify documents with AI-powered analysis and RSA-signed certificates,
            verify certification integrity, and compare documents using
            spectral frequency analysis.
          </motion.p>

        </div>
      </section>

      <div className="h-px bg-gradient-to-r from-transparent via-blue-500/20 to-transparent" />
      <section className="max-w-6xl mx-auto px-6 pt-16 pb-20">
        <motion.div
          className="text-center mb-20"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h2 className="text-2xl sm:text-3xl font-bold text-base-color mb-3">Features</h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Three integrated tools for forensic document certification,
            verification, and comparison.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto [&>*:last-child]:md:col-span-2 [&>*:last-child]:md:max-w-[calc(50%-1rem)] [&>*:last-child]:md:mx-auto">
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              custom={i}
              variants={fadeUp}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
            >
              <Link to={feature.path} className="block h-full">
                <div
                  className={`relative h-full rounded-xl border ${feature.border} bg-gradient-to-br ${feature.color} p-6 transition-all duration-300 hover:shadow-lg hover:shadow-black/20 group`}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className={`p-2.5 rounded-lg bg-black/20 ${feature.accent}`}>
                      <feature.icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-xl font-semibold text-base-color">
                      {feature.title}
                    </h3>
                    <ArrowRight className="w-4 h-4 text-muted-foreground ml-auto opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                  </div>

                  <p className="text-sm md:text-base text-muted-foreground leading-relaxed mb-4">
                    {feature.description}
                  </p>

                  <div className="border-t border-white/5 pt-4 mb-4">
                    <p className="text-xs font-medium text-base-color mb-1">How to use</p>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {feature.howTo}
                    </p>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {feature.details.map((detail) => (
                      <span
                        key={detail}
                        className="text-[10px] px-2 py-1 rounded-full bg-white/5 text-muted-foreground border border-white/5"
                      >
                        {detail}
                      </span>
                    ))}
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="border-t border-white/5 py-20">
        <div className="max-w-5xl mx-auto px-6">
          <motion.div
            className="text-center mb-12"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="text-2xl sm:text-3xl font-bold text-base-color mb-3">
              Design Principles
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Built with security, transparency, and scientific rigor at every layer.
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {principles.map((p, i) => (
              <motion.div
                key={p.title}
                custom={i}
                variants={fadeUp}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="w-12 h-12 mx-auto mb-4 rounded-xl bg-blue-500/10 flex items-center justify-center">
                  <p.icon className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-sm font-semibold text-base-color mb-2">
                  {p.title}
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {p.text}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-t border-white/5 py-16">
        <div className="max-w-3xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <FileText className="w-10 h-10 mx-auto mb-4 text-blue-400/60" />
            <h3 className="text-2xl font-bold text-base-color mb-3">
              Ready to get started?
            </h3>
            <p className="text-muted-foreground mb-6">
              Upload a document to analyze, certify, verify, or compare. No account required for analysis, verification, or comparison.
            </p>
            <Link to="/sign">
              <Button size="lg" className="bg-blue-600 hover:bg-blue-700 text-white">
                Get Started
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
