describe("Navigation", () => {
  it("loads home page (Physical)", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")
  })

  it("nav links work", () => {
    cy.visit("/")
    cy.contains("Verify").click()
    cy.url().should("include", "/verify")

    cy.contains("OCR").click()
    cy.url().should("include", "/ocr")

    cy.contains("About").click()
    cy.url().should("include", "/about")

    cy.contains("Analyze").click()
    cy.url().should("eq", Cypress.config("baseUrl") + "/")
  })

  it("shows sign-in button when not authenticated", () => {
    cy.clearLocalStorage()
    cy.visit("/")
    cy.contains("Sign in").should("be.visible")
  })

  it("sign page requires auth", () => {
    cy.clearLocalStorage()
    cy.visit("/sign")
    cy.url().should("include", "/auth")
  })
})
