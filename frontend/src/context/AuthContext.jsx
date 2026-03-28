import React, { createContext, useContext, useMemo } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  return <AuthContext.Provider value={null}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const {
    isAuthenticated,
    isLoading,
    user,
    loginWithRedirect,
    logout: auth0Logout,
    getAccessTokenSilently,
  } = useAuth0();

  return useMemo(() => ({
    isAuthenticated,
    isInitializing: isLoading,
    user: user ? {
      id: user.sub,
      email: user.email,
      fullName: user.name,
      picture: user.picture,
    } : null,
    token: null,
    login: () => loginWithRedirect(),
    signup: () => loginWithRedirect({ authorizationParams: { screen_hint: 'signup' } }),
    logout: () => auth0Logout({ logoutParams: { returnTo: window.location.origin } }),
    getToken: getAccessTokenSilently,
  }), [isAuthenticated, isLoading, user, loginWithRedirect, auth0Logout, getAccessTokenSilently]);
}
