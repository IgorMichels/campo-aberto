(() => {
  "use strict";

  // Pairs with the inline anti-flash script in each page's <head>, which
  // already applies any stored preference to <html data-theme="..."> before
  // first paint. This file only wires up the button: toggling the
  // attribute, persisting the choice, and keeping the icon in sync.
  const STORAGE_KEY = "campo-aberto-theme";
  const button = document.getElementById("theme-toggle");
  if (!button) return;

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  const currentTheme = () =>
    document.documentElement.getAttribute("data-theme") || (prefersDark.matches ? "dark" : "light");

  const applyIcon = (theme) => {
    const isDark = theme === "dark";
    button.textContent = isDark ? "☀️" : "🌙";
    button.setAttribute("aria-pressed", String(isDark));
    button.setAttribute("aria-label", isDark ? "Mudar para tema claro" : "Mudar para tema escuro");
  };

  applyIcon(currentTheme());

  button.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem(STORAGE_KEY, next);
    applyIcon(next);
  });

  // No explicit choice stored yet (no data-theme attribute) -- follow the OS
  // preference live instead of freezing whatever it was at page load.
  prefersDark.addEventListener("change", () => {
    if (!document.documentElement.hasAttribute("data-theme")) {
      applyIcon(currentTheme());
    }
  });
})();
