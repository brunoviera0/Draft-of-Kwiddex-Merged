describe("CNN Scoring", () => {
  it("uploads image and shows Monte Carlo results", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")

    cy.fixture("test-document.png", null).then((img) => {
      cy.get('input[type="file"]').first().selectFile(
        { contents: img, fileName: "test.png", mimeType: "image/png" },
        { force: true }
      )
    })

    cy.contains("button", "Check document").click()

    // Wait for Monte Carlo response (30 samples can take time)
    cy.contains("Confidence", { timeout: 60000 }).should("be.visible")

    // Should show percentage
    cy.contains(/\d+%/).should("exist")

    // Should show CI bounds
    cy.contains("95% Confidence Interval").should("be.visible")

    // Should show Monte Carlo stats
    cy.contains("Monte Carlo").should("be.visible")
    cy.contains("Samples").should("be.visible")
    cy.contains("Agreement").should("be.visible")

    // Should show model info
    cy.contains("resnet18").should("be.visible")
  })

  it("can reset after scoring", () => {
    cy.visit("/")

    cy.fixture("test-document.png", null).then((img) => {
      cy.get('input[type="file"]').first().selectFile(
        { contents: img, fileName: "test.png", mimeType: "image/png" },
        { force: true }
      )
    })

    cy.contains("button", "Check document").click()
    cy.contains("Confidence", { timeout: 60000 }).should("be.visible")
    cy.contains("button", "Reset").click()
  })
})
