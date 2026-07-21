// GoatCounter page-view tracking (https://campo-aberto.goatcounter.com).
// Injected as a real <script> tag rather than duplicated in every page's
// <head>, since every page already loads this file (see site/README.md's
// "Analytics" section for why GoatCounter was picked).
(function () {
  var script = document.createElement("script");
  script.async = true;
  script.src = "//gc.zgo.at/count.js";
  script.setAttribute("data-goatcounter", "https://campo-aberto.goatcounter.com/count");
  document.head.appendChild(script);
})();
