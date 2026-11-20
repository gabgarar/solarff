// MathJax configuration: only the spans written by the arithmatex extension are typeset.
// MathJax itself is served from this site (js/tex-svg.js), so the formulas do not depend on an external CDN.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  svg: { fontCache: "global" }
};

// Typeset again after every page change (Material loads pages without a full reload when instant navigation is on)
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
