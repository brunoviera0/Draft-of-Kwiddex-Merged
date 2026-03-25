import React, { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';

export default function AuthGuard({ children }) {
  const { isAuthenticated, isInitializing, login } = useAuth();

  useEffect(() => {
    if (!isInitializing && !isAuthenticated) {
      login();
    }
  }, [isInitializing, isAuthenticated, login]);

  if (isInitializing) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-foreground">
        <p className="text-sm text-muted-foreground">Loading your session...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-foreground">
        <p className="text-sm text-muted-foreground">Redirecting to login...</p>
      </div>
    );
  }

  return children;
}
