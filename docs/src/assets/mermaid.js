// Render Mermaid flowcharts inside Documenter.jl pages.
//
// Documenter.jl outputs ```mermaid blocks as:
//   <pre><code class="language-mermaid">…diagram source…</code></pre>
//
// This script loads Mermaid.js from the jsDelivr CDN, locates those elements,
// swaps them for <div class="mermaid"> containers, and calls mermaid.run().
//
// Documenter.jl embeds RequireJS, whose global `define` is detected by
// Mermaid's UMD bundle.  When that happens the bundle registers itself as an
// AMD module instead of setting window.mermaid, so the onload callback fails.
// We temporarily hide `define` while the CDN script loads to force the bundle
// into browser-global mode.

(function () {
    "use strict";

    function init() {
        // Hide RequireJS's define so Mermaid's UMD bundle falls through to
        // browser-global mode and sets window.mermaid.
        var savedDefine = window.define;
        window.define = undefined;

        var script = document.createElement("script");
        script.type = "text/javascript";
        // Pin to a specific minor version for reproducibility.
        script.src =
            "https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js";
        script.onload = function () {
            window.define = savedDefine; // restore RequireJS
            mermaid.initialize({
                startOnLoad: false,
                theme: "default",
                securityLevel: "loose",
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                },
            });
            renderAll();
        };
        document.head.appendChild(script);
    }

    function renderAll() {
        // Documenter wraps every code block in <pre><code class="language-X">.
        var blocks = document.querySelectorAll(
            "pre > code.language-mermaid"
        );
        if (!blocks.length) return;

        blocks.forEach(function (code) {
            var source = code.textContent;
            var pre    = code.parentElement;

            var wrapper = document.createElement("div");
            wrapper.style.textAlign = "center";
            wrapper.style.margin    = "1.5rem 0";
            wrapper.style.overflowX = "auto";

            var div = document.createElement("div");
            div.className   = "mermaid";
            div.textContent = source;

            wrapper.appendChild(div);
            pre.parentElement.replaceChild(wrapper, pre);
        });

        mermaid.run();
    }

    // Fire after the DOM is fully parsed.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
