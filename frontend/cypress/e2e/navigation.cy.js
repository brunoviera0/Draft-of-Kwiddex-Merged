describe("Navigation", () => {
  it("loads home page with Kwiddex branding", () => {
    cy.visit("/")
    cy.contains("Kwiddex").should("be.visible")
    cy.contains("forensic document analysis").should("be.visible")
  })

  it("nav links to Analyze work", () => {
    cy.visit("/")
    cy.contains("Analyze").click()
    cy.url().should("include", "/analyze")
    cy.contains("Physical document checks").should("be.visible")
  })

  it("nav links to Verify work", () => {
    cy.visit("/")
    cy.contains("a", "Verify").click()
    cy.url().should("include", "/verify")
  })

  it("nav links to Compare work", () => {
    cy.visit("/")
    cy.contains("a", "Compare").click()
    cy.url().should("include", "/compare")
  })

  it("nav links to About work", () => {
    cy.visit("/")
    cy.contains("a", "About").click()
    cy.url().should("include", "/about")
  })

  it("home page feature cards link to correct pages", () => {
    cy.visit("/")
    cy.contains("Start Analyzing").click()
    cy.url().should("include", "/analyze")
  })

  it("compare page shows LWSP description", () => {
    cy.visit("/compare")
    cy.contains("How it works").should("be.visible")
  })

  it("unknown routes redirect to home", () => {
    cy.visit("/nonexistent")
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })
})
