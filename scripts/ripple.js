const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")
let reduceMotion = prefersReducedMotion?.matches ?? false

const updatePreference = (event) => {
  reduceMotion = event.matches
}

if (prefersReducedMotion?.addEventListener) {
  prefersReducedMotion.addEventListener("change", updatePreference)
} else if (prefersReducedMotion?.addListener) {
  prefersReducedMotion.addListener(updatePreference)
}

const removeRipple = (ripple) => {
  if (!ripple) return
  ripple.remove()
}

const createRipple = (target, event) => {
  if (reduceMotion) return

  const rect = target.getBoundingClientRect()
  const ripple = document.createElement("span")
  ripple.className = "motion-ripple"

  const maxDimension = Math.max(rect.width, rect.height)
  const size = maxDimension * 2
  const x = (event?.clientX ?? rect.left + rect.width / 2) - rect.left - size / 2
  const y = (event?.clientY ?? rect.top + rect.height / 2) - rect.top - size / 2

  ripple.style.setProperty("--motion-ripple-size", `${size}px`)
  ripple.style.setProperty("--motion-ripple-x", `${x}px`)
  ripple.style.setProperty("--motion-ripple-y", `${y}px`)

  const fill = target.getAttribute("data-ripple-color")
  if (fill) {
    ripple.style.setProperty("--motion-ripple-fill", fill)
  }

  target.appendChild(ripple)

  ripple.addEventListener("animationend", () => removeRipple(ripple), { once: true })
}

const pointerHandler = (event) => {
  if (event.button !== undefined && event.button !== 0) return

  const target = event.target?.closest?.("[data-ripple]")
  if (!target) return
  if (target.matches?.(":disabled, [aria-disabled='true']")) return

  createRipple(target, event)
}

document.addEventListener("pointerdown", pointerHandler, { passive: true })

document.addEventListener("keydown", (event) => {
  if (event.key !== " " && event.key !== "Enter") return

  const target = event.target?.closest?.("[data-ripple]")
  if (!target) return
  if (target.matches?.(":disabled, [aria-disabled='true']")) return

  createRipple(target)
})

window.addEventListener("blur", () => {
  document.querySelectorAll?.(".motion-ripple").forEach((node) => removeRipple(node))
})

export {}
