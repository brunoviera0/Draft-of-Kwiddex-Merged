describe("Auth0 Authentication", () => {
  beforeEach(() => {
    cy.clearLocalStorage()
  })

  it("shows Sign in button when not logged in", () => {
    cy.visit("/")
    cy.contains("Sign in").should("be.visible")
    cy.contains("Log out").should("not.exist")
  })

  it("Sign in button exists and is clickable", () => {
    cy.visit("/")
    cy.contains("Sign in").should("be.visible").and("not.be.disabled")
  })

  it("Sign page is accessible without login", () => {
    cy.visit("/sign")
    cy.url().should("include", "/sign")
    cy.contains("Sign in required").should("be.visible")
  })

  it("public pages do not require auth", () => {
    cy.visit("/")
    cy.contains("Kwiddex").should("be.visible")

    cy.visit("/verify")
    cy.url().should("include", "/verify")

    cy.visit("/compare")
    cy.contains("Scales of Justice").should("be.visible")

    cy.visit("/sign")
    cy.url().should("include", "/sign")

    cy.visit("/about")
    cy.url().should("include", "/about")
  })
})
