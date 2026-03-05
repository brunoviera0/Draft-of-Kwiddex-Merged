const DEFAULT_FALLBACK_MESSAGE = "An unexpected error occurred.";

const singleLine = (value) => value.replace(/\s+/g, " ").trim();

const safeStringify = (value) => {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch (error) {
    return String(value);
  }
};

export function normalizeError(input, options = {}) {
  const {
    fallbackTitle = "Error",
    fallbackMessage = DEFAULT_FALLBACK_MESSAGE,
    status,
    debug,
  } = options;

  let statusCode = Number.isFinite(Number(status)) ? Number(status) : undefined;
  const isErrorInstance = input instanceof Error;
  const isPlainObject =
    typeof input === "object" && input !== null && !Array.isArray(input) && !isErrorInstance;

  const jsonPayload = isPlainObject ? input : null;

  if (jsonPayload && Number.isFinite(Number(jsonPayload.status))) {
    statusCode = Number(jsonPayload.status);
  }

  if (!statusCode && isErrorInstance && Number.isFinite(Number(input.status))) {
    statusCode = Number(input.status);
  }

  let messageCandidate = "";

  if (jsonPayload && jsonPayload.details !== undefined) {
    messageCandidate = safeStringify(jsonPayload.details);
  }

  if (!messageCandidate && jsonPayload && typeof jsonPayload.error === "string") {
    messageCandidate = jsonPayload.error;
  }

  if (!messageCandidate && jsonPayload && typeof jsonPayload.message === "string") {
    messageCandidate = jsonPayload.message;
  }

  if (!messageCandidate && isErrorInstance && typeof input.message === "string") {
    messageCandidate = input.message;
  }

  if (!messageCandidate && typeof input === "string") {
    messageCandidate = input;
  }

  if (!messageCandidate) {
    messageCandidate = fallbackMessage;
  }

  const message = singleLine(String(messageCandidate || fallbackMessage));

  let detailSource = debug;

  if (!detailSource && jsonPayload) {
    detailSource = safeStringify(jsonPayload);
  }

  if (!detailSource && typeof input === "string") {
    detailSource = input;
  }

  if (!detailSource && isErrorInstance) {
    detailSource = safeStringify(input.stack || input.message);
  }

  if (!detailSource) {
    detailSource = message;
  }

  const result = {
    title: fallbackTitle,
    message,
  };

  const detailsString = safeStringify(detailSource);
  if (detailsString) {
    result.details = detailsString;
  }

  if (Number.isFinite(statusCode) && statusCode !== 200) {
    result.status = statusCode;
  }

  return result;
}

export default normalizeError;
