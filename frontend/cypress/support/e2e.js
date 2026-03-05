const API = "http://localhost:3001"

Cypress.Commands.add("apiSignup", (email, password) => {
  return cy.request({
    method: "POST",
    url: `${API}/api/auth/signup`,
    body: { email, password },
    failOnStatusCode: false,
  })
})

Cypress.Commands.add("apiLogin", (email, password) => {
  return cy.request({
    method: "POST",
    url: `${API}/api/auth/login`,
    body: { email, password },
  }).then((res) => {
    window.localStorage.setItem("kwiddex.auth", JSON.stringify({
      token: res.body.token,
      user: res.body.user,
    }))
  })
})
