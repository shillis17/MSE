document.addEventListener("DOMContentLoaded", async () => {
  try {
    const rootData = await loadRootInfo();
    const availableModels = Array.isArray(rootData.available_models) ? rootData.available_models : [];
    const defaultModel = rootData.default_model || "panns";

    setText("defaultModelStat", defaultModel.toUpperCase());
    setText("modelsStat", availableModels.length ? availableModels.map((m) => m.toUpperCase()).join(", ") : defaultModel.toUpperCase());
  } catch (error) {
    console.error(error);
  }
});
