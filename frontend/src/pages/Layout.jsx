import { Link, Outlet, useLocation } from 'react-router-dom'
import { Shield, ScanLine, FileCheck, Newspaper, Fingerprint, Info } from 'lucide-react'
import { useAuth } from '@/context/AuthContext'

const NAV_ITEMS = [
  { to: '/', label: 'Analyze', icon: ScanLine },
  { to: '/verify', label: 'Verify', icon: FileCheck },
  { to: '/ocr', label: 'OCR', icon: Newspaper },
  { to: '/sign', label: 'Sign', icon: Fingerprint },
  { to: '/about', label: 'About', icon: Info },
]

export default function Layout() {
  const { user, isAuthenticated, login, logout } = useAuth()
  const location = useLocation()

  return (
    <div className="min-h-screen bg-background text-foreground">
      <nav className="sticky top-0 z-40 border-b border-border bg-card/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
          <Link to="/" className="inline-flex items-center gap-2 font-semibold">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Shield className="h-4 w-4" />
            </span>
            Kwiddex
          </Link>

          <div className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const isActive = location.pathname === to
              return (
                <Link
                  key={to}
                  to={to}
                  className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              )
            })}
          </div>

          <div className="flex items-center gap-3">
            {isAuthenticated ? (
              <>
                <span className="hidden text-sm text-muted-foreground md:inline">
                  {user?.fullName || user?.email}
                </span>
                <button
                  type="button"
                  onClick={logout}
                  className="text-sm font-medium text-muted-foreground hover:text-foreground"
                >
                  Log out
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={login}
                className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                Sign in
              </button>
            )}
          </div>
        </div>
      </nav>

      <main>
        <Outlet />
      </main>

      <footer className="border-t border-border bg-card">
        <div className="mx-auto max-w-6xl px-4 py-6 text-center text-xs text-muted-foreground">
          CNN-based analysis should be reviewed alongside policy and human verification.
        </div>
      </footer>
    </div>
  )
}
