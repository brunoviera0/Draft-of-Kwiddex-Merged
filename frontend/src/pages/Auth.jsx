import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { useAuth } from '@/context/AuthContext';

export default function AuthPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, signup } = useAuth();

  const redirectTo = location.state?.from || '/';

  const [mode, setMode] = useState('login');
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const submitLabel = mode === 'login' ? 'Log in' : 'Create account';

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setError('');

    try {
      if (mode === 'login') {
        await login({ email, password });
      } else {
        await signup({ fullName, email, password });
      }
      navigate(redirectTo, { replace: true });
    } catch (requestError) {
      setError(requestError.message || 'Unable to authenticate.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-10 text-foreground">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-2 text-center">
          <div className="mx-auto inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <Shield className="h-5 w-5" />
          </div>
          <CardTitle>{mode === 'login' ? 'Welcome back' : 'Create your account'}</CardTitle>
          <CardDescription>
            {mode === 'login' ? 'Log in to access the documentation tools.' : 'Sign up to start verifying documents.'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-3" onSubmit={handleSubmit}>
            {mode === 'signup' && (
              <Input
                placeholder="Full name"
                value={fullName}
                onChange={(event) => setFullName(event.target.value)}
                required
                disabled={submitting}
              />
            )}
            <Input
              type="email"
              placeholder="Email address"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              disabled={submitting}
            />
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={8}
              disabled={submitting}
            />

            {error && <p className="text-sm text-destructive">{error}</p>}

            <Button className="w-full" type="submit" disabled={submitting}>
              {submitting ? 'Please wait...' : submitLabel}
            </Button>
          </form>

          <p className="mt-4 text-center text-sm text-muted-foreground">
            {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
            <button
              type="button"
              className="font-medium text-primary underline-offset-2 hover:underline"
              onClick={() => setMode(mode === 'login' ? 'signup' : 'login')}
              disabled={submitting}
            >
              {mode === 'login' ? 'Sign up' : 'Log in'}
            </button>
          </p>

          <p className="mt-3 text-center text-xs text-muted-foreground">
            <Link className="hover:underline" to="/">Return to home</Link>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
