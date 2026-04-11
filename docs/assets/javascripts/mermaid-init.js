(() => {
  function mermaidTheme() {
    return document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default";
  }

  function upgradeMermaidBlocks() {
    const blocks = document.querySelectorAll("pre.mermaid");
    blocks.forEach((block) => {
      const container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = block.textContent || "";
      block.replaceWith(container);
    });
  }

  async function renderMermaid() {
    if (!window.mermaid) {
      return;
    }

    upgradeMermaidBlocks();
    window.mermaid.initialize({
      startOnLoad: false,
      theme: mermaidTheme(),
      securityLevel: "loose",
      flowchart: {
        htmlLabels: true,
        curve: "basis"
      }
    });

    try {
      await window.mermaid.run({
        querySelector: ".mermaid"
      });
    } catch (error) {
      console.error("Mermaid render failed", error);
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(() => {
      renderMermaid();
    });
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  } else {
    renderMermaid();
  }
})();
