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

  it("Sign page triggers Auth0 redirect", () => {
    cy.visit("/sign", { failOnStatusCode: false })
    cy.origin("https://dev-jamm61acuiu8yfq6.us.auth0.com", () => {
      cy.url().should("include", "auth0.com")
    })
  })

  it("public pages do not require auth", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")

    cy.visit("/verify")
    cy.url().should("include", "/verify")

    cy.visit("/compare")
    cy.contains("Scales of Justice").should("be.visible")

    cy.visit("/about")
    cy.url().should("include", "/about")
  })
})
