import { defineConfig } from "cypress"

export default defineConfig({
  e2e: {
    baseUrl: "https://kwiddex.com",
    supportFile: "cypress/support/e2e.js",
    specPattern: "cypress/e2e/**/*.cy.{js,jsx}",
    viewportWidth: 1280,
    viewportHeight: 720,
    defaultCommandTimeout: 15000,
    video: false,
    chromeWebSecurity: false,
  },
})
