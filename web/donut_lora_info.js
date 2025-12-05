import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Donut LoRA Info Display Extension
// Shows CivitAI metadata directly in the node when a LoRA is selected

// Block weight presets - uses "NAME:vector" format like Inspire
// The vector part after ':' is extracted and used

app.registerExtension({
    name: "donut.LoRAInfoDisplay",

    // Register settings in ComfyUI settings panel
    async setup() {
        // CivitAI API Key setting
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.ApiKey",
            name: "[DonutNodes] CivitAI API Key",
            type: "text",
            defaultValue: "",
            tooltip: "Your CivitAI API key for fetching LoRA metadata. Get one from: https://civitai.com/user/account -> API Keys",
            attrs: {
                style: {
                    fontFamily: "monospace",
                },
                type: "password",  // Hide the API key
            },
            async onChange(value) {
                // Save to server config
                try {
                    await api.fetchApi("/donut/config/civitai_api_key", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ api_key: value }),
                    });
                    console.log("[DonutNodes] CivitAI API key saved");
                } catch (error) {
                    console.error("[DonutNodes] Error saving API key:", error);
                }
            },
        });

        // Auto-lookup toggle
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.AutoLookup",
            name: "[DonutNodes] Auto-fetch LoRA info from CivitAI",
            type: "boolean",
            defaultValue: true,
            tooltip: "Automatically fetch LoRA metadata from CivitAI when selected",
            async onChange(value) {
                try {
                    await api.fetchApi("/donut/config/civitai_auto_lookup", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ auto_lookup: value }),
                    });
                } catch (error) {
                    console.error("[DonutNodes] Error saving setting:", error);
                }
            },
        });

        // Download previews toggle
        app.ui.settings.addSetting({
            id: "DonutNodes.CivitAI.DownloadPreviews",
            name: "[DonutNodes] Download LoRA preview images",
            type: "boolean",
            defaultValue: true,
            tooltip: "Download and cache preview images from CivitAI",
            async onChange(value) {
                try {
                    await api.fetchApi("/donut/config/civitai_download_previews", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ download_previews: value }),
                    });
                } catch (error) {
                    console.error("[DonutNodes] Error saving setting:", error);
                }
            },
        });

        // Load current settings from server
        try {
            const response = await api.fetchApi("/donut/config");
            if (response.ok) {
                const config = await response.json();
                // Settings will be loaded by ComfyUI from localStorage,
                // but we sync with server config on startup
                if (config.civitai?.api_key) {
                    app.ui.settings.setSettingValue("DonutNodes.CivitAI.ApiKey", config.civitai.api_key);
                }
            }
        } catch (error) {
            console.log("[DonutNodes] Could not load server config:", error);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Hook into DonutLoRAStack node
        if (nodeData.name === "DonutLoRAStack") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Find the block_preset widget and add callback
                const presetWidget = this.widgets?.find(w => w.name === "block_preset");
                if (presetWidget) {
                    const originalCallback = presetWidget.callback;
                    const node = this;

                    presetWidget.callback = function(value) {
                        // Call original callback if exists
                        if (originalCallback) {
                            originalCallback.call(this, value);
                        }

                        // Extract the vector part after ':' (Inspire-style format)
                        // Format is "NAME:vector" e.g. "SDXL-ALL:1,1,1,1,1,1,1,1,1,1,1,1,1"
                        if (value && value !== "None" && value.includes(":")) {
                            const presetVector = value.split(":")[1];
                            for (const w of node.widgets) {
                                if (w.name === "block_vector_1" ||
                                    w.name === "block_vector_2" ||
                                    w.name === "block_vector_3") {
                                    w.value = presetVector;
                                }
                            }
                            node.setDirtyCanvas(true);
                        }
                    };
                }

                // Create 3 pairs of info text + image display (one per LoRA slot)
                this.infoWidgets = [];
                this.imageWidgets = [];

                for (let i = 1; i <= 3; i++) {
                    // Text info widget
                    const infoWidget = ComfyWidgets["STRING"](
                        this,
                        `civitai_info_${i}`,
                        ["STRING", { multiline: true }],
                        app
                    ).widget;

                    infoWidget.inputEl.readOnly = true;
                    infoWidget.inputEl.style.opacity = "0.9";
                    infoWidget.inputEl.style.fontSize = "10px";
                    infoWidget.inputEl.style.fontFamily = "monospace";
                    infoWidget.inputEl.style.backgroundColor = "#1a1a2e";
                    infoWidget.inputEl.style.color = "#eee";
                    infoWidget.inputEl.style.minHeight = "50px";
                    infoWidget.inputEl.placeholder = `LoRA ${i} info...`;
                    infoWidget.serializeValue = async () => "";
                    this.infoWidgets.push(infoWidget);

                    // Image widget for collage
                    const imgWidget = this.addDOMWidget(`civitai_image_${i}`, "image", document.createElement("div"), {
                        serialize: false,
                    });
                    imgWidget.element.style.display = "flex";
                    imgWidget.element.style.justifyContent = "center";
                    imgWidget.element.style.alignItems = "center";
                    imgWidget.element.style.backgroundColor = "#1a1a2e";
                    imgWidget.element.style.minHeight = "128px";
                    imgWidget.element.style.borderRadius = "4px";
                    imgWidget.element.style.overflow = "hidden";
                    imgWidget.computeSize = () => [this.size[0] - 20, 128];
                    this.imageWidgets.push(imgWidget);
                }

                // Adjust node size for 3 text+image pairs
                this.setSize([this.size[0], this.size[1] + 550]);
            };

            // Update display when node executes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                // Update individual info widgets
                // message.text contains [lora_info, trigger_words, urls, info_1, info_2, info_3]
                if (this.infoWidgets && message.text) {
                    const info1 = message.text[3] || "";
                    const info2 = message.text[4] || "";
                    const info3 = message.text[5] || "";

                    this.infoWidgets[0].value = info1 || "LoRA 1: Off or None";
                    this.infoWidgets[1].value = info2 || "LoRA 2: Off or None";
                    this.infoWidgets[2].value = info3 || "LoRA 3: Off or None";
                }

                // Update image widgets with collages
                if (this.imageWidgets && message.images) {
                    // message.images is [slot0, slot1, slot2] - may contain null for empty slots
                    for (let i = 0; i < 3; i++) {
                        // Clear the widget
                        this.imageWidgets[i].element.innerHTML = "";

                        const imgData = message.images[i];
                        if (imgData && imgData.filename) {
                            const img = document.createElement("img");
                            img.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&subfolder=${encodeURIComponent(imgData.subfolder || "")}&type=${imgData.type || "temp"}`);
                            img.style.maxWidth = "100%";
                            img.style.maxHeight = "128px";
                            img.style.objectFit = "contain";
                            img.style.borderRadius = "4px";
                            this.imageWidgets[i].element.appendChild(img);
                        } else {
                            // Show placeholder for empty slot
                            const placeholder = document.createElement("div");
                            placeholder.style.color = "#666";
                            placeholder.style.fontSize = "11px";
                            placeholder.textContent = `No preview for LoRA ${i + 1}`;
                            this.imageWidgets[i].element.appendChild(placeholder);
                        }
                    }

                    // Remove images from message so default handler doesn't show them again
                    delete message.images;
                }

                // Call original handler (for any other functionality)
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Trigger resize
                this.setDirtyCanvas(true);
            };
        }

        // Hook into DonutLoRACivitAIInfo node for detailed info display
        if (nodeData.name === "DonutLoRACivitAIInfo") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Create info display
                const infoWidget = ComfyWidgets["STRING"](
                    this,
                    "info_display",
                    ["STRING", { multiline: true }],
                    app
                ).widget;

                infoWidget.inputEl.readOnly = true;
                infoWidget.inputEl.style.fontSize = "11px";
                infoWidget.inputEl.style.fontFamily = "monospace";
                infoWidget.inputEl.style.backgroundColor = "#1a1a2e";
                infoWidget.inputEl.style.minHeight = "100px";
                infoWidget.inputEl.placeholder = "Select a LoRA to see CivitAI info...";
                infoWidget.serializeValue = async () => "";

                this.infoWidget = infoWidget;
                this.setSize([this.size[0], this.size[1] + 80]);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                if (this.infoWidget && message.text) {
                    const [name, version, description, triggerWords, modelUrl, hash, recWeight] = message.text;

                    let display = "";
                    if (name) {
                        display = `Name: ${name}`;
                        if (version) display += ` (${version})`;
                        display += "\n";
                        if (recWeight && recWeight !== "1.0") {
                            display += `Recommended Weight: ${recWeight}\n`;
                        }
                        if (triggerWords) {
                            display += `Triggers: ${triggerWords}\n`;
                        }
                        if (description) {
                            display += `\n${description.substring(0, 200)}${description.length > 200 ? '...' : ''}`;
                        }
                        if (modelUrl) {
                            display += `\n\n${modelUrl}`;
                        }
                    } else {
                        display = "Select a LoRA to see info...";
                    }

                    this.infoWidget.value = display;
                }
            };
        }
    },
});
