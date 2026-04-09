describe("Navigation", () => {
  it("loads home page (Analyze)", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")
  })

  it("nav links to Verify work", () => {
    cy.visit("/")
    cy.contains("Verify").click()
    cy.url().should("include", "/verify")
  })

  it("nav links to Compare work", () => {
    cy.visit("/")
    cy.contains("Compare").click()
    cy.url().should("include", "/compare")
  })

  it("nav links to About work", () => {
    cy.visit("/")
    cy.contains("About").click()
    cy.url().should("include", "/about")
  })

  it("nav links back to Analyze work", () => {
    cy.visit("/about")
    cy.contains("Analyze").click()
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })

  it("shows sign in button when not authenticated", () => {
    cy.clearLocalStorage()
    cy.visit("/")
    cy.contains("Sign in").should("be.visible")
  })

  it("verify page loads and shows upload", () => {
    cy.visit("/verify")
    cy.contains("Upload PDF").should("be.visible")
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
    cy.url().should("include", "/about")
  })

  it("unknown routes redirect to home", () => {
    cy.visit("/nonexistent")
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })
})
