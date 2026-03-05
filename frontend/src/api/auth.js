import { API_BASE } from '@/api/verify';

const AUTH_STORAGE_KEY = 'kwiddex.auth';

export const loadAuthState = () => {
  if (typeof window === 'undefined') return null;
  const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (!raw) return null;

  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
};

export const saveAuthState = (state) => {
  if (typeof window === 'undefined') return;

  if (!state) {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }

  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(state));
};

const joinUrl = (base, path) => {
  const safeBase = (base || '').replace(/\/+$/, '');
  const safePath = path.startsWith('/') ? path : `/${path}`;
  return `${safeBase}${safePath}`;
};

const parseErrorResponse = async (response) => {
  const responseText = await response.text();

  if (!responseText) {
    return null;
  }

  try {
    const parsed = JSON.parse(responseText);
    return parsed?.error || parsed?.message || null;
  } catch {
    if (typeof responseText === 'string' && responseText.trim().startsWith('<')) {
      if (import.meta.env.DEV) {
        console.error('Auth endpoint returned HTML instead of JSON:', responseText);
      }
      return null;
    }

    return responseText.trim() || null;
  }
};

const request = async (path, { method = 'GET', body, token } = {}) => {
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  const endpoint = joinUrl(API_BASE, `/auth${path}`);

  let response;

  try {
    response = await fetch(endpoint, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new Error('Unable to reach the authentication service. Please try again later.');
  }

  if (response.ok) {
    return response.json().catch(() => ({}));
  }

  const responseError = await parseErrorResponse(response);
  const fallbackMessage = response.status === 404
    ? 'Signup service is unavailable at the configured URL. Please verify API base settings.'
    : 'Authentication request failed. Please try again.';

  throw new Error(responseError || fallbackMessage);
};

export const signup = (payload) => request('/signup', { method: 'POST', body: payload });
export const login = (payload) => request('/login', { method: 'POST', body: payload });
export const getCurrentUser = (token) => request('/me', { token });
