describe("Document Analysis", () => {
  it("shows upload area on certify page", () => {
    cy.visit("/sign")
    cy.contains("Certify Document").should("be.visible")
    cy.contains("Upload PDF").should("be.visible")
    cy.get('input[type="file"]').should("exist")
  })
  it("compare page shows upload panels", () => {
    cy.visit("/compare")
    cy.contains("Scales of Justice").should("be.visible")
    cy.contains("Upload image").should("be.visible")
  })
})
