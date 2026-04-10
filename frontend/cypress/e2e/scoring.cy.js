describe("CNN Scoring", () => {
  it("uploads image and shows results", () => {
    cy.visit("/analyze")
    cy.contains("Physical document checks").should("be.visible")

    cy.fixture("test-document.png", null).then((img) => {
      cy.get('input[type="file"]').first().selectFile(
        { contents: img, fileName: "test.png", mimeType: "image/png" },
        { force: true }
      )
    })

    cy.contains("button", "Check document").click()
    cy.contains("Confidence", { timeout: 90000 }).should("be.visible")
    cy.contains("CNN Analysis", { timeout: 10000 }).should("be.visible")
  })

  it("can reset after scoring", () => {
    cy.visit("/analyze")

    cy.fixture("test-document.png", null).then((img) => {
      cy.get('input[type="file"]').first().selectFile(
        { contents: img, fileName: "test.png", mimeType: "image/png" },
        { force: true }
      )
    })

    cy.contains("button", "Check document").click()
    cy.contains("Confidence", { timeout: 90000 }).should("be.visible")
    cy.contains("button", "Reset").click()
  })
})
