import { useState, useEffect } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { Home as HomeIcon, FileCheck, Scale, Fingerprint, Info, Menu, X } from 'lucide-react'
import KwiddexLogo from '@/assets/Kwiddex_logo.png'
import { useAuth } from '@/context/AuthContext'

const NAV_ITEMS = [
  { to: '/', label: 'Home', icon: HomeIcon },
  { to: '/sign', label: 'Certify', icon: Fingerprint },
  { to: '/verify', label: 'Verify', icon: FileCheck },
  { to: '/compare', label: 'Compare', icon: Scale },
  { to: '/about', label: 'About', icon: Info },
]

export default function Layout() {
  const { user, isAuthenticated, login, logout } = useAuth()
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <nav className="sticky top-0 z-40 border-b border-border bg-card/95 backdrop-blur">
        <div className="mx-auto flex h-20 max-w-6xl items-center justify-between px-4">
          <Link to="/" className="inline-flex items-center shrink-0">
            <img src={KwiddexLogo} alt="Kwiddex" className="h-12 w-auto" />
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const isActive = location.pathname === to
              return (
                <Link
                  key={to}
                  to={to}
                  className={`inline-flex items-center gap-1.5 rounded-md px-3.5 py-2 text-base font-medium transition-colors ${
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {label}
                </Link>
              )
            })}
          </div>

          <div className="flex items-center gap-2">
            {isAuthenticated ? (
              <>
                <span className="hidden text-xs text-muted-foreground lg:inline">
                  {user?.fullName || user?.email}
                </span>
                <button
                  type="button"
                  onClick={logout}
                  className="text-xs font-medium text-muted-foreground hover:text-foreground bg-transparent border-none shadow-none p-0 focus:outline-none focus:ring-0 active:bg-transparent"
                >
                  Log out
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={login}
                className="rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
              >
                Sign in
              </button>
            )}

            {/* Mobile hamburger */}
            <button
              type="button"
              className="md:hidden p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted active:bg-muted"
              onClick={() => setMobileOpen(prev => !prev)}
            >
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        {/* Mobile dropdown */}
        {mobileOpen && (
          <div className="md:hidden border-t border-border bg-card px-2 py-2">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
              const isActive = location.pathname === to
              return (
                <Link
                  key={to}
                  to={to}
                  className={`flex items-center gap-3 rounded-md px-4 py-3.5 text-base font-medium transition-colors active:bg-muted ${
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  {label}
                </Link>
              )
            })}
          </div>
        )}
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
