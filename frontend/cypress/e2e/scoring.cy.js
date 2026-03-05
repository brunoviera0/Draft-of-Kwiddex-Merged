describe("CNN Scoring", () => {
  it("uploads image and shows confidence + CI", () => {
    cy.visit("/")
    cy.contains("Physical document checks").should("be.visible")

    cy.fixture("test-document.png", null).then((img) => {
      cy.get('input[type="file"]').first().selectFile(
        { contents: img, fileName: "test.png", mimeType: "image/png" },
        { force: true }
      )
    })

    cy.contains("button", "Check document").click()

    // Wait for CNN response
    cy.contains("Confidence", { timeout: 30000 }).should("be.visible")

    // Should show percentage
    cy.contains(/\d+\.\d+%/).should("exist")

    // Should show CI bounds
    cy.contains("95% Confidence Interval").should("be.visible")

    // Should NOT show old AiResult fields
    cy.contains("Likelihood original").should("not.exist")
    cy.contains("Reasons").should("not.exist")
    cy.contains("Flags").should("not.exist")
    cy.contains("/ 100").should("not.exist")
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
    cy.contains("Confidence", { timeout: 30000 }).should("be.visible")
    cy.contains("button", "Reset").click()
    cy.contains("Run the scorer to see results.").should("be.visible")
  })
})
