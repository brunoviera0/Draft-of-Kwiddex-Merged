describe("Verify Page", () => {
  it("loads with upload prompt", () => {
    cy.visit("/verify")
    cy.contains("Upload PDF").should("be.visible")
  })

  it("uploads PDF and checks for certificate", () => {
    cy.visit("/verify")

    const pdfContent = "%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    const blob = new Blob([pdfContent], { type: "application/pdf" })

    cy.get('input[type="file"]').first().selectFile(
      { contents: blob, fileName: "test.pdf", mimeType: "application/pdf" },
      { force: true }
    )

    cy.contains("test.pdf", { timeout: 10000 }).should("be.visible")
  })

  it("shows certificate check result", () => {
    cy.visit("/verify")

    const pdfContent = "%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
    const blob = new Blob([pdfContent], { type: "application/pdf" })

    cy.get('input[type="file"]').first().selectFile(
      { contents: blob, fileName: "test.pdf", mimeType: "application/pdf" },
      { force: true }
    )

    // Should show some result about certificate status
    cy.get('[class*="card"], [class*="Card"]', { timeout: 15000 }).should("exist")
  })
})
