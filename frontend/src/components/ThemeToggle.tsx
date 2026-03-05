import { useEffect, useState } from "react";
import { getInitialTheme, setTheme, toggleTheme } from "@/lib/theme";

export default function ThemeToggle() {
  const [theme, setLocal] = useState<"light" | "dark">("light");

  useEffect(() => {
    const t = getInitialTheme();
    setLocal(t);
    // react to OS changes if user hasn't picked manually
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => {
      const saved = localStorage.getItem("kwiddex:theme");
      if (!saved) {
        const sys = mq && mq.matches ? "dark" : "light";
        setTheme(sys);
        setLocal(sys);
      }
    };
    mq?.addEventListener?.("change", onChange);
    return () => mq?.removeEventListener?.("change", onChange);
  }, []);

  function handleClick() {
    toggleTheme();
    const next = document.documentElement.getAttribute("data-theme") as "light" | "dark";
    setLocal(next);
  }

  const label = theme === "dark" ? "Light" : "Dark";

  return (
    <button
      onClick={handleClick}
      className="theme-toggle btn-animated"
      aria-label="Toggle color theme"
      aria-pressed={theme === "dark"}
      title={"Switch to " + label + " mode"}
    >
      {/* simple icon pair */}
      <span aria-hidden>{theme === "dark" ? "🌞" : "🌙"}</span>
    </button>
  );
}
