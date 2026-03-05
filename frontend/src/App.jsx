import './App.css'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import Layout from '@/pages/Layout'
import Physical from '@/pages/Physical'
import VerifyPage from '@/pages/Verify'
import OcrPage from '@/pages/Ocr'
import SignPage from '@/pages/Sign'
import About from '@/pages/About'
import AuthPage from '@/pages/Auth'
import ThemeToggle from '@/components/ThemeToggle'

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Physical embedded />} />
            <Route path="/verify" element={<VerifyPage />} />
            <Route path="/ocr" element={<OcrPage />} />
            <Route path="/sign" element={<SignPage />} />
            <Route path="/about" element={<About />} />
          </Route>
          <Route path="/auth" element={<AuthPage />} />

          {/* Legacy redirects */}
          <Route path="/physical" element={<Navigate to="/" replace />} />
          <Route path="/dashboard" element={<Navigate to="/" replace />} />
          <Route path="/Dashboard" element={<Navigate to="/" replace />} />
          <Route path="/digital/*" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        <Toaster />
      </Router>
      <ThemeToggle />
    </>
  )
}

export default App
