describe("Auth", () => {
  const email = `cy-${Date.now()}@test.kwiddex.com`
  const password = "CypressTest123!"

  it("shows login form", () => {
    cy.visit("/auth")
    cy.contains("Welcome back")
    cy.get('input[type="email"]').should("be.visible")
    cy.get('input[type="password"]').should("be.visible")
  })

  it("signs up and redirects home", () => {
    cy.visit("/auth")
    cy.contains("Sign up").click()
    cy.get('input[placeholder="Full name"]').type("Cypress User")
    cy.get('input[type="email"]').type(email)
    cy.get('input[type="password"]').type(password)
    cy.contains("Create account").click()
    cy.url({ timeout: 10000 }).should("eq", Cypress.config("baseUrl") + "/")
  })

  it("logs in and redirects home", () => {
    cy.clearLocalStorage()
    cy.visit("/auth")
    cy.get('input[type="email"]').type(email)
    cy.get('input[type="password"]').type(password)
    cy.contains("button", "Log in").click()
    cy.url({ timeout: 10000 }).should("eq", Cypress.config("baseUrl") + "/")
    cy.contains("Log out").should("be.visible")
  })

  it("rejects wrong password", () => {
    cy.visit("/auth")
    cy.get('input[type="email"]').type(email)
    cy.get('input[type="password"]').type("WrongPass!")
    cy.contains("button", "Log in").click()
    cy.get(".text-destructive", { timeout: 5000 }).should("be.visible")
  })
})
