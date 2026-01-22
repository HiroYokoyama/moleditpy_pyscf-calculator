document.addEventListener("DOMContentLoaded", function () {
    // Select all code blocks that have bash or powershell language classes
    const codeBlocks = document.querySelectorAll("pre code.language-bash, pre code.language-powershell");

    codeBlocks.forEach(function (codeBlock) {
        const pre = codeBlock.parentNode;

        // Check if already wrapped
        if (pre.parentNode.classList.contains("code-wrapper")) {
            return;
        }

        // Create a wrapper for the pre element
        const wrapper = document.createElement("div");
        wrapper.className = "code-wrapper";
        wrapper.style.position = "relative";

        // Insert wrapper before pre
        pre.parentNode.insertBefore(wrapper, pre);

        // Move pre inside wrapper
        wrapper.appendChild(pre);

        // Create the button
        const button = document.createElement("button");
        button.className = "copy-btn";
        button.textContent = "Copy";
        button.title = "Copy to clipboard";

        // Add click event
        button.addEventListener("click", function () {
            // Get text content (innerText respects block display newlines and excludes pseudo-elements)
            const textToCopy = codeBlock.innerText;

            navigator.clipboard.writeText(textToCopy).then(function () {
                // Success feedback
                const originalText = button.textContent;
                button.textContent = "Copied!";
                button.classList.add("copied");

                setTimeout(function () {
                    button.textContent = originalText;
                    button.classList.remove("copied");
                }, 2000);
            }).catch(function (err) {
                console.error("Failed to copy text: ", err);
                button.textContent = "Error";
            });
        });

        // Append button to the wrapper (not the pre)
        wrapper.appendChild(button);
    });
});
