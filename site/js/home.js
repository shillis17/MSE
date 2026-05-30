document.addEventListener("DOMContentLoaded", async () => {
  try {
    const rootData = await loadRootInfo();
    const availableModels = normalizeModelOptions(rootData.available_models);
    const defaultModel = rootData.default_model || "panns";

    setText("defaultModelStat", defaultModel.toUpperCase());
    setText("modelsStat", availableModels.length ? availableModels.map((modelOption) => modelOption.label).join(", ") : defaultModel.toUpperCase());
  } catch (error) {
    console.error(error);
  }
});
