import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { getCurrentUser, loadAuthState, login, saveAuthState, signup } from '@/api/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [authState, setAuthState] = useState(() => loadAuthState());
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      if (!authState?.token) {
        setIsInitializing(false);
        return;
      }

      try {
        const { user } = await getCurrentUser(authState.token);
        if (cancelled) return;
        const nextState = { token: authState.token, user };
        setAuthState(nextState);
        saveAuthState(nextState);
      } catch {
        if (cancelled) return;
        setAuthState(null);
        saveAuthState(null);
      } finally {
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    };

    initialize();
    return () => {
      cancelled = true;
    };
  }, []);

  const actions = useMemo(() => ({
    async login(credentials) {
      const result = await login(credentials);
      const nextState = { token: result.token, user: result.user };
      setAuthState(nextState);
      saveAuthState(nextState);
      return result;
    },
    async signup(payload) {
      const result = await signup(payload);
      const nextState = { token: result.token, user: result.user };
      setAuthState(nextState);
      saveAuthState(nextState);
      return result;
    },
    logout() {
      setAuthState(null);
      saveAuthState(null);
    },
  }), []);

  const value = useMemo(() => ({
    token: authState?.token ?? null,
    user: authState?.user ?? null,
    isAuthenticated: Boolean(authState?.token),
    isInitializing,
    ...actions,
  }), [actions, authState, isInitializing]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider.');
  }
  return context;
}
