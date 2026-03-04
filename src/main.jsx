import React from 'react'
import ReactDOM from 'react-dom/client'
import App from '@/App.jsx'
import { AuthProvider } from '@/context/AuthContext'
import '../styles/motion.css'
import '@/index.css'
import '@/styles/theme.css'
import '../scripts/ripple.js'

ReactDOM.createRoot(document.getElementById('root')).render(
  <AuthProvider>
    <App />
  </AuthProvider>
)
