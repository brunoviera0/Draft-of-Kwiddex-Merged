const LOCAL_HOSTNAMES = ['localhost', '127.0.0.1', '::1', '0.0.0.0'];

const isLocalHostname = (hostname = '') => {
  const normalised = hostname.toLowerCase();
  if (LOCAL_HOSTNAMES.includes(normalised)) {
    return true;
  }

  // Catch other loopback variations such as 127.*
  return normalised.startsWith('127.');
};

const removeTrailingSlash = (value) => value?.replace(/\/$/, '') ?? value;

const resolveConfiguredBase = (value) => {
  if (!value) return null;

  if (typeof window === 'undefined') {
    return removeTrailingSlash(value.trim());
  }

  try {
    const parsed = new URL(value, window.location.origin);
    const basePath = removeTrailingSlash(parsed.pathname);
    return `${parsed.origin}${basePath === '/' ? '' : basePath}`;
  } catch (error) {
    console.warn('Invalid VITE_API_BASE value. Falling back to defaults.', error);
    return null;
  }
};

const getDefaultApiBase = () => {
  const configuredBase = resolveConfiguredBase(import.meta.env.VITE_API_BASE);

  if (configuredBase && typeof window !== 'undefined') {
    try {
      const configuredHostname = new URL(configuredBase).hostname;
      const currentHostname = window.location.hostname;

      if (isLocalHostname(configuredHostname) && !isLocalHostname(currentHostname)) {
        // Avoid leaking a localhost-only base URL into production builds.
      } else {
        return configuredBase;
      }
    } catch {
      // If parsing fails here, fall through to the runtime defaults.
    }
  } else if (configuredBase) {
    return configuredBase;
  }

  if (typeof window !== 'undefined') {
    const { origin, hostname } = window.location;

    if (isLocalHostname(hostname)) {
      return 'http://localhost:3001';
    }

    return origin;
  }

  return '';
};

export const API_BASE = getDefaultApiBase();

export async function verifyDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  let response;

  try {
    response = await fetch(`${API_BASE}/api/verify`, {
      method: 'POST',
      body: formData
    });
  } catch (networkError) {
    console.error('Network error while verifying PDF:', networkError);
    throw new Error('Unable to reach the verification service. Please try again later.');
  }

  if (!response.ok) {
    let errorMessage = 'Failed to verify the PDF. Please try again.';
    try {
      const errorBody = await response.json();
      if (errorBody?.error) {
        errorMessage = errorBody.error;
      }
    } catch {
      // Ignore parsing errors and fall back to default message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}
