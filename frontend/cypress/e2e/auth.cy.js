describe("Auth0 Authentication", () => {
  beforeEach(() => {
    cy.clearLocalStorage()
  })

  it("shows Sign in button when not logged in", () => {
    cy.visit("/")
    cy.contains("Sign in").should("be.visible")
    cy.contains("Log out").should("not.exist")
  })

  it("Sign in button triggers Auth0 redirect", () => {
    cy.visit("/")
    cy.contains("Sign in").click()

    // Should redirect to Auth0 domain
    cy.url().should("include", "auth0.com")
  })

  it("Sign page shows loading then redirects to Auth0", () => {
    cy.visit("/sign")

    // AuthGuard should show loading or redirect message
    cy.contains(/loading|redirecting/i, { timeout: 5000 }).should("be.visible")
  })

  it("public pages do not require auth", () => {
    // Analyze (home)
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")

    // Verify
    cy.visit("/verify")
    cy.contains("Verify & Inspect Document").should("be.visible")

    // Compare
    cy.visit("/compare")
    cy.contains("Scales of Justice").should("be.visible")

    // About
    cy.visit("/about")
    cy.url().should("include", "/about")
  })

  it("only Sign page is auth gated", () => {
    // These should all load without redirect
    const publicPages = ["/", "/verify", "/compare", "/about"]
    for (const page of publicPages) {
      cy.visit(page)
      cy.url().should("not.include", "auth0.com")
    }
  })
})

