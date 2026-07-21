(() => {
  "use strict";

  // Wires up every .nav-dropdown in the top nav (currently just "Jogos") --
  // shared across all 6 pages (index/probabilities/evolution/matches/*.html)
  // since the nav itself is site-wide chrome, not page-specific logic. Hovering
  // already reveals the menu via pure CSS (:hover); this only adds the
  // click/tap path (mouse click toggles it open/closed, a second click
  // outside or Escape closes it) for touch devices, which can't hover.
  document.querySelectorAll(".nav-dropdown").forEach((dropdown) => {
    const trigger = dropdown.querySelector(".nav-dropdown-trigger");
    if (!trigger) return;

    const setOpen = (open) => {
      dropdown.classList.toggle("is-open", open);
      trigger.setAttribute("aria-expanded", String(open));
    };

    // The trigger stays a real <a href="...upcoming.html"> so it still
    // navigates somewhere sensible if this script fails to load; only
    // intercept the click once we know the dropdown is actually wired up.
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      setOpen(!dropdown.classList.contains("is-open"));
    });

    document.addEventListener("click", (event) => {
      if (!dropdown.contains(event.target)) setOpen(false);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setOpen(false);
    });
  });
})();
