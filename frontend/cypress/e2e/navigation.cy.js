describe("Navigation", () => {
  it("loads home page with Kwiddex branding", () => {
    cy.visit("/")
    cy.contains("Kwiddex").should("be.visible")
    cy.contains("forensic document analysis").should("be.visible")
  })
  it("nav links to Certify work", () => {
    cy.visit("/")
    cy.contains("a", "Certify").click()
    cy.url().should("include", "/sign")
    cy.contains("Certify Document").should("be.visible")
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
  it("compare page shows description", () => {
    cy.visit("/compare")
    cy.contains("multi-region forensic analysis").should("be.visible")
  })
  it("unknown routes redirect to home", () => {
    cy.visit("/nonexistent")
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })
})
