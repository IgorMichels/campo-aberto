// PLACEHOLDER -- no analytics provider is wired up yet.
//
// This file is intentionally inert. It exists as the single place to add a
// privacy-conscious page-view/traffic snippet once a provider is chosen (see
// site/README.md's "Analytics" section for the tradeoffs and a suggested
// default). Do NOT duplicate a vendor's boilerplate <script> tag across every
// page -- add it here once, since every page already loads this file.
//
// To activate, replace this file's contents with the provider's snippet, e.g.:
//
//   Plausible / GoatCounter (script-tag-only, no config object needed):
//     <script defer data-domain="igormichels.github.io" src="https://plausible.io/js/script.js"></script>
//     -> move that <script> tag into each page's <head> instead of this file,
//        or keep using this file as a loader if the provider offers a JS API.
//
//   Google Analytics (GA4, needs a real Measurement ID -- G-XXXXXXXXXX):
//     window.dataLayer = window.dataLayer || [];
//     function gtag(){dataLayer.push(arguments);}
//     gtag("js", new Date());
//     gtag("config", "G-XXXXXXXXXX");
//     -> also add <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
//        to each page's <head>.
//
// Until then, this file deliberately does nothing.
