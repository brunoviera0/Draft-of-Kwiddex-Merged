export type Theme = "light" | "dark";
const KEY = "kwiddex:theme";

export function getSystemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function getInitialTheme(): Theme {
  const saved = typeof window !== "undefined" ? (localStorage.getItem(KEY) as Theme | null) : null;
  return saved ?? getSystemTheme();
}

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.setAttribute("data-theme", theme);
  // Helps form controls & scrollbars on supported UAs
  root.style.colorScheme = theme;
  document.documentElement.classList.toggle("dark", theme === "dark");
}

export function setTheme(theme: Theme) {
  localStorage.setItem(KEY, theme);
  applyTheme(theme);
}

export function toggleTheme() {
  const current = (document.documentElement.getAttribute("data-theme") as Theme) || getSystemTheme();
  setTheme(current === "dark" ? "light" : "dark");
}
