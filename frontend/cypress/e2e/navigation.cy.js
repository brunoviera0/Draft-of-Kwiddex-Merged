describe("Navigation", () => {
  it("loads home page (Analyze)", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")
  })

  it("nav links work in correct order", () => {
    cy.visit("/")

    cy.contains("Sign").click()
    cy.url().should("include", "/sign")

    cy.contains("Verify").click()
    cy.url().should("include", "/verify")

    cy.contains("Compare").click()
    cy.url().should("include", "/compare")

    cy.contains("About").click()
    cy.url().should("include", "/about")

    cy.contains("Analyze").click()
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })

  it("shows sign in button when not authenticated", () => {
    cy.clearLocalStorage()
    cy.visit("/")
    cy.contains("Sign in").should("be.visible")
  })

  it("sign page redirects to Auth0 login", () => {
    cy.clearLocalStorage()
    cy.visit("/sign")
    cy.contains("Redirecting to login").should("be.visible")
  })

  it("verify page loads and shows upload", () => {
    cy.visit("/verify")
    cy.contains("Verify & Inspect Document").should("be.visible")
  })

  it("compare page loads with two tabs", () => {
    cy.visit("/compare")
    cy.contains("Scales of Justice").should("be.visible")
    cy.contains("Compare").should("be.visible")
    cy.contains("Extract Text").should("be.visible")
  })

  it("compare page shows LWSP description", () => {
    cy.visit("/compare")
    cy.contains("How it works").should("be.visible")
    cy.contains("Linear Wave Stochastic Process").should("be.visible")
  })

  it("about page loads", () => {
    cy.visit("/about")
    cy.contains("About").should("be.visible")
  })

  it("unknown routes redirect to home", () => {
    cy.visit("/nonexistent")
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })
})
