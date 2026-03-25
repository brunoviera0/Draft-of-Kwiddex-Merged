import { useEffect } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { Shield } from 'lucide-react';

export default function AuthPage() {
  const { isAuthenticated, isInitializing, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = location.state?.from || '/';

  useEffect(() => {
    if (isInitializing) return;
    if (isAuthenticated) {
      navigate(redirectTo, { replace: true });
      return;
    }
    login();
  }, [isAuthenticated, isInitializing, login, navigate, redirectTo]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-10 text-foreground">
      <div className="text-center space-y-4">
        <div className="mx-auto inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <Shield className="h-5 w-5" />
        </div>
        <p className="text-sm text-muted-foreground">Redirecting to login...</p>
      </div>
    </div>
  );
}
