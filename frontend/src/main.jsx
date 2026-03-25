import React from 'react'
import ReactDOM from 'react-dom/client'
import { Auth0Provider } from '@auth0/auth0-react'
import App from '@/App.jsx'
import '../styles/motion.css'
import '@/index.css'
import '@/styles/theme.css'
import '../scripts/ripple.js'

const AUTH0_DOMAIN = 'dev-jamm61acuiu8yfq6.us.auth0.com'
const AUTH0_CLIENT_ID = 'yBA5UqZj65KlgdyHsmQs4RycS4JqN8jf'
const AUTH0_AUDIENCE = 'https://api.kwiddex.com'

ReactDOM.createRoot(document.getElementById('root')).render(
  <Auth0Provider
    domain={AUTH0_DOMAIN}
    clientId={AUTH0_CLIENT_ID}
    authorizationParams={{
      redirect_uri: window.location.origin,
      audience: AUTH0_AUDIENCE,
    }}
    cacheLocation="localstorage"
  >
    <App />
  </Auth0Provider>
)
