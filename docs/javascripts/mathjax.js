window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"]
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"]
    ]
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex",
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};
