import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Donut LoRA Browser
 * A popup dialog to browse, preview, and select LoRAs with CivitAI info
 */

class DonutLoraBrowser {
    constructor() {
        this.allLoras = [];     // Full list
        this.loras = [];        // Filtered list
        this.currentIndex = 0;
        this.currentInfo = null;
        this.isLoading = false;
        this.dialog = null;
        this.onSelect = null;
        this.imageIndex = 0;    // For cycling through individual images
        this.filterText = "";   // Search filter
    }

    async loadLoraList() {
        try {
            const response = await api.fetchApi("/donut/loras/list");
            if (response.ok) {
                const data = await response.json();
                this.allLoras = data.loras || [];
                this.applyFilter();
                return true;
            }
        } catch (error) {
            console.error("[DonutNodes] Error loading LoRA list:", error);
        }
        return false;
    }

    applyFilter() {
        if (!this.filterText) {
            this.loras = [...this.allLoras];
        } else {
            const filter = this.filterText.toLowerCase();
            this.loras = this.allLoras.filter(l =>
                l.name.toLowerCase().includes(filter) ||
                l.filename.toLowerCase().includes(filter)
            );
        }
        // Clamp current index
        if (this.currentIndex >= this.loras.length) {
            this.currentIndex = Math.max(0, this.loras.length - 1);
        }
    }

    async loadLoraInfo(loraName) {
        this.isLoading = true;
        this.updateUI();

        try {
            const response = await api.fetchApi(`/donut/loras/info?name=${encodeURIComponent(loraName)}`);
            if (response.ok) {
                const data = await response.json();
                this.currentInfo = data;
                this.imageIndex = 0;
            } else {
                this.currentInfo = { name: loraName, error: "Failed to load info" };
            }
        } catch (error) {
            console.error("[DonutNodes] Error loading LoRA info:", error);
            this.currentInfo = { name: loraName, error: error.message };
        }

        this.isLoading = false;
        this.updateUI();
    }

    createDialog() {
        // Create overlay
        const overlay = document.createElement("div");
        overlay.id = "donut-lora-browser-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create dialog
        const dialog = document.createElement("div");
        dialog.id = "donut-lora-browser-dialog";
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
            min-width: 700px;
            max-width: 900px;
            max-height: 85vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            color: #eee;
            font-family: sans-serif;
        `;

        // Header with title and counter
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        `;

        const title = document.createElement("h2");
        title.textContent = "LoRA Browser";
        title.style.cssText = "margin: 0; font-size: 18px;";

        const counter = document.createElement("span");
        counter.id = "donut-lora-counter";
        counter.style.cssText = "color: #888; font-size: 14px;";

        header.appendChild(title);
        header.appendChild(counter);
        dialog.appendChild(header);

        // Search box
        const searchContainer = document.createElement("div");
        searchContainer.style.cssText = `
            margin-bottom: 15px;
        `;

        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.id = "donut-lora-search";
        searchInput.placeholder = "Search LoRAs...";
        searchInput.style.cssText = `
            width: 100%;
            padding: 8px 12px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 13px;
            box-sizing: border-box;
        `;
        searchInput.oninput = (e) => {
            this.filterText = e.target.value;
            this.applyFilter();
            this.currentIndex = 0;
            this.imageIndex = 0;
            if (this.loras.length > 0) {
                this.loadLoraInfo(this.loras[0].name);
            } else {
                this.currentInfo = null;
                this.updateUI();
            }
        };
        // Prevent arrow keys from triggering navigation while typing
        searchInput.onkeydown = (e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight" ||
                e.key === "ArrowUp" || e.key === "ArrowDown") {
                e.stopPropagation();
            }
            if (e.key === "Escape") {
                searchInput.blur();
                e.stopPropagation();
            }
        };

        searchContainer.appendChild(searchInput);
        dialog.appendChild(searchContainer);

        // Navigation
        const nav = document.createElement("div");
        nav.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        `;

        const prevBtn = document.createElement("button");
        prevBtn.textContent = "< Prev";
        prevBtn.id = "donut-lora-prev";
        prevBtn.style.cssText = this.getButtonStyle();
        prevBtn.onclick = () => this.navigate(-1);

        const loraName = document.createElement("div");
        loraName.id = "donut-lora-name";
        loraName.style.cssText = `
            flex: 1;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 0 10px;
        `;

        const nextBtn = document.createElement("button");
        nextBtn.textContent = "Next >";
        nextBtn.id = "donut-lora-next";
        nextBtn.style.cssText = this.getButtonStyle();
        nextBtn.onclick = () => this.navigate(1);

        nav.appendChild(prevBtn);
        nav.appendChild(loraName);
        nav.appendChild(nextBtn);
        dialog.appendChild(nav);

        // Content area (image + info)
        const content = document.createElement("div");
        content.style.cssText = `
            display: flex;
            gap: 20px;
            flex: 1;
            overflow: hidden;
        `;

        // Image area
        const imageArea = document.createElement("div");
        imageArea.style.cssText = `
            width: 300px;
            min-width: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        `;

        const imageContainer = document.createElement("div");
        imageContainer.id = "donut-lora-image";
        imageContainer.style.cssText = `
            width: 100%;
            height: 300px;
            background: #0d0d1a;
            border-radius: 4px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        `;

        // Image navigation buttons
        const imageNav = document.createElement("div");
        imageNav.style.cssText = `
            display: flex;
            gap: 10px;
        `;

        const prevImgBtn = document.createElement("button");
        prevImgBtn.textContent = "< Img";
        prevImgBtn.style.cssText = this.getButtonStyle(true);
        prevImgBtn.onclick = () => this.navigateImage(-1);

        const imgCounter = document.createElement("span");
        imgCounter.id = "donut-lora-img-counter";
        imgCounter.style.cssText = "color: #888; font-size: 12px; padding: 5px;";

        const nextImgBtn = document.createElement("button");
        nextImgBtn.textContent = "Img >";
        nextImgBtn.style.cssText = this.getButtonStyle(true);
        nextImgBtn.onclick = () => this.navigateImage(1);

        imageNav.appendChild(prevImgBtn);
        imageNav.appendChild(imgCounter);
        imageNav.appendChild(nextImgBtn);

        imageArea.appendChild(imageContainer);
        imageArea.appendChild(imageNav);
        content.appendChild(imageArea);

        // Info area
        const infoArea = document.createElement("div");
        infoArea.id = "donut-lora-info";
        infoArea.style.cssText = `
            flex: 1;
            overflow-y: auto;
            font-size: 12px;
            line-height: 1.5;
        `;
        content.appendChild(infoArea);

        dialog.appendChild(content);

        // Footer with hints and buttons
        const footer = document.createElement("div");
        footer.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
        `;

        const hints = document.createElement("div");
        hints.style.cssText = "color: #666; font-size: 11px;";
        hints.innerHTML = "<kbd>←</kbd><kbd>→</kbd> LoRAs &nbsp; <kbd>↑</kbd><kbd>↓</kbd> Images &nbsp; <kbd>Enter</kbd> Select &nbsp; <kbd>Esc</kbd> Close";

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 10px;";

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = this.getButtonStyle();
        cancelBtn.onclick = () => this.close();

        const selectBtn = document.createElement("button");
        selectBtn.textContent = "Select LoRA";
        selectBtn.id = "donut-lora-select";
        selectBtn.style.cssText = this.getButtonStyle() + "background: #4a6fa5;";
        selectBtn.onclick = () => this.selectCurrent();

        buttons.appendChild(cancelBtn);
        buttons.appendChild(selectBtn);
        footer.appendChild(hints);
        footer.appendChild(buttons);
        dialog.appendChild(footer);

        overlay.appendChild(dialog);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) this.close();
        };

        // Keyboard navigation
        overlay.tabIndex = 0;
        overlay.onkeydown = (e) => {
            if (e.key === "ArrowLeft") {
                this.navigate(-1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.navigate(1);
                e.preventDefault();
            } else if (e.key === "ArrowUp") {
                this.navigateImage(-1);
                e.preventDefault();
            } else if (e.key === "ArrowDown") {
                this.navigateImage(1);
                e.preventDefault();
            } else if (e.key === "Enter") {
                this.selectCurrent();
                e.preventDefault();
            } else if (e.key === "Escape") {
                this.close();
                e.preventDefault();
            }
        };

        this.dialog = overlay;
        return overlay;
    }

    getButtonStyle(small = false) {
        return `
            background: #333;
            color: #eee;
            border: 1px solid #555;
            border-radius: 4px;
            padding: ${small ? "5px 10px" : "8px 15px"};
            cursor: pointer;
            font-size: ${small ? "11px" : "13px"};
            transition: background 0.2s;
        `;
    }

    navigate(direction) {
        if (this.loras.length === 0) return;

        this.currentIndex += direction;
        if (this.currentIndex < 0) this.currentIndex = this.loras.length - 1;
        if (this.currentIndex >= this.loras.length) this.currentIndex = 0;

        this.imageIndex = 0;
        this.loadLoraInfo(this.loras[this.currentIndex].name);
    }

    navigateImage(direction) {
        if (!this.currentInfo?.civitai?.images) return;

        const imageCount = this.currentInfo.civitai.images.length;
        if (imageCount <= 1) return;

        this.imageIndex += direction;
        if (this.imageIndex < 0) this.imageIndex = imageCount - 1;
        if (this.imageIndex >= imageCount) this.imageIndex = 0;

        this.updateImage();
    }

    updateUI() {
        if (!this.dialog) return;

        const counter = this.dialog.querySelector("#donut-lora-counter");
        const nameEl = this.dialog.querySelector("#donut-lora-name");
        const imageEl = this.dialog.querySelector("#donut-lora-image");
        const infoEl = this.dialog.querySelector("#donut-lora-info");

        // Handle empty list
        if (this.loras.length === 0) {
            counter.textContent = "0 / 0";
            nameEl.textContent = "No matches";
            nameEl.title = "";
            if (imageEl) imageEl.innerHTML = '<div style="color: #888;">No LoRAs match your search</div>';
            if (infoEl) infoEl.innerHTML = '<div style="color: #888; text-align: center; padding: 20px;">Try a different search term</div>';
            return;
        }

        // Update counter
        counter.textContent = `${this.currentIndex + 1} / ${this.loras.length}`;

        // Update name
        if (this.loras[this.currentIndex]) {
            nameEl.textContent = this.loras[this.currentIndex].name;
            nameEl.title = this.loras[this.currentIndex].name;
        }

        // Update image
        this.updateImage();

        // Update info
        if (this.isLoading) {
            infoEl.innerHTML = '<div style="text-align: center; padding: 20px;">Loading...</div>';
        } else if (this.currentInfo) {
            infoEl.innerHTML = this.formatInfo(this.currentInfo);
        }
    }

    updateImage() {
        const imageEl = this.dialog?.querySelector("#donut-lora-image");
        const imgCounterEl = this.dialog?.querySelector("#donut-lora-img-counter");
        if (!imageEl) return;

        if (this.isLoading) {
            imageEl.innerHTML = '<div style="color: #888;">Loading...</div>';
            if (imgCounterEl) imgCounterEl.textContent = "";
            return;
        }

        if (this.currentInfo?.hash) {
            // Show individual image based on imageIndex
            const img = document.createElement("img");
            const imageType = this.imageIndex === 0 ? "0" : String(this.imageIndex);
            img.src = api.apiURL(`/donut/loras/preview?hash=${this.currentInfo.hash}&type=${imageType}&t=${Date.now()}`);
            img.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain;";
            img.onerror = () => {
                // Try collage as fallback
                img.src = api.apiURL(`/donut/loras/preview?hash=${this.currentInfo.hash}&type=collage&t=${Date.now()}`);
                img.onerror = () => {
                    imageEl.innerHTML = '<div style="color: #888;">No preview</div>';
                };
            };
            imageEl.innerHTML = "";
            imageEl.appendChild(img);

            // Update image counter
            if (imgCounterEl && this.currentInfo?.civitai?.images) {
                const total = Math.min(this.currentInfo.civitai.images.length, 4);
                imgCounterEl.textContent = `${this.imageIndex + 1} / ${total}`;
            } else if (imgCounterEl) {
                imgCounterEl.textContent = "";
            }
        } else {
            imageEl.innerHTML = '<div style="color: #888;">No preview</div>';
            if (imgCounterEl) imgCounterEl.textContent = "";
        }
    }

    formatInfo(info) {
        if (info.error && !info.civitai) {
            return `<div style="color: #f88;">${info.error}</div>`;
        }

        const c = info.civitai;
        if (!c) {
            return `
                <div style="color: #888;">
                    <p>No CivitAI info available for this LoRA.</p>
                    <p style="font-size: 11px; margin-top: 10px;">Hash: ${info.hash || "Unknown"}</p>
                </div>
            `;
        }

        let html = "";

        // Model name and version
        html += `<div style="font-size: 14px; font-weight: bold; margin-bottom: 10px;">
            ${c.model_name || "Unknown"}
            <span style="font-weight: normal; color: #888;">(${c.version_name || ""})</span>
        </div>`;

        // Base model
        if (c.base_model) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Base Model:</span>
                <span style="background: #333; padding: 2px 6px; border-radius: 3px;">${c.base_model}</span>
            </div>`;
        }

        // Recommended weight
        if (c.recommended_weight && c.recommended_weight !== 1.0) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Recommended Weight:</span> ${c.recommended_weight}
            </div>`;
        }

        // Trigger words
        if (c.trained_words && c.trained_words.length > 0) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Trigger Words:</span><br>
                <div style="background: #0d0d1a; padding: 8px; border-radius: 4px; margin-top: 4px; font-family: monospace; font-size: 11px;">
                    ${c.trained_words.join(", ")}
                </div>
            </div>`;
        }

        // Stats
        html += `<div style="margin-bottom: 8px; color: #888; font-size: 11px;">
            Downloads: ${c.download_count?.toLocaleString() || 0}
            ${c.rating ? ` | Rating: ${c.rating.toFixed(1)}` : ""}
        </div>`;

        // Description (stripped of HTML, truncated)
        if (c.description) {
            const desc = c.description.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
            const truncated = desc.length > 300 ? desc.substring(0, 300) + "..." : desc;
            html += `<div style="margin-bottom: 8px; color: #aaa; font-size: 11px;">
                ${truncated}
            </div>`;
        }

        // CivitAI link
        if (c.model_url) {
            html += `<div style="margin-top: 10px;">
                <a href="${c.model_url}" target="_blank" style="color: #6a9fd4; text-decoration: none; font-size: 11px;">
                    View on CivitAI
                </a>
            </div>`;
        }

        return html;
    }

    selectCurrent() {
        if (this.loras[this.currentIndex] && this.onSelect) {
            this.onSelect(this.loras[this.currentIndex].name);
        }
        this.close();
    }

    close() {
        if (this.dialog) {
            this.dialog.remove();
            this.dialog = null;
        }
    }

    async show(onSelect, initialLora = null) {
        this.onSelect = onSelect;
        this.filterText = "";  // Reset filter
        this.imageIndex = 0;

        // Load LoRA list
        const loaded = await this.loadLoraList();
        if (!loaded || this.loras.length === 0) {
            alert("No LoRAs found!");
            return;
        }

        // Find initial index if provided
        if (initialLora) {
            const idx = this.loras.findIndex(l => l.name === initialLora);
            if (idx >= 0) this.currentIndex = idx;
        } else {
            this.currentIndex = 0;
        }

        // Create and show dialog
        const dialog = this.createDialog();
        document.body.appendChild(dialog);
        dialog.focus();

        // Load initial info
        this.loadLoraInfo(this.loras[this.currentIndex].name);
    }
}

// Create global instance
const loraBrowser = new DonutLoraBrowser();

// Export for use in other modules
window.DonutLoraBrowser = loraBrowser;

// Register extension to add context menu option
app.registerExtension({
    name: "donut.LoraBrowser",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DonutLoRAStack") {
            // Add right-click menu option
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }

                // Add separator
                options.push(null);

                // Add browse option for each LoRA slot
                for (let i = 1; i <= 3; i++) {
                    const slotNum = i;
                    options.push({
                        content: `Browse LoRAs (Slot ${slotNum})`,
                        callback: () => {
                            const loraWidget = this.widgets?.find(w => w.name === `lora_name_${slotNum}`);
                            const currentLora = loraWidget?.value;

                            loraBrowser.show((selectedLora) => {
                                if (loraWidget) {
                                    loraWidget.value = selectedLora;
                                    // Also enable the slot
                                    const switchWidget = this.widgets?.find(w => w.name === `switch_${slotNum}`);
                                    if (switchWidget) {
                                        switchWidget.value = "On";
                                    }
                                    this.setDirtyCanvas(true);
                                }
                            }, currentLora !== "None" ? currentLora : null);
                        }
                    });
                }
            };
        }
    }
});
