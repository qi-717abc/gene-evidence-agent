(() => {
  const STORAGE_KEY = "gea.education.preferences.v1";
  const FONT_SIZES = new Set(["standard", "large", "largest"]);
  const DEFAULTS = Object.freeze({ favoriteIds: [], fontSize: "standard" });

  function normalize(value) {
    if (!value || typeof value !== "object" || Array.isArray(value) || value.version !== 1) {
      return { favoriteIds: [], recentIds: [], fontSize: DEFAULTS.fontSize };
    }
    const normalizeIds = (ids, limit) => Array.isArray(ids)
      ? [...new Set(ids.filter((id) => typeof id === "string" && /^K-edu-[a-z0-9-]{1,80}$/.test(id)))].slice(0, limit)
      : [];
    return {
      favoriteIds: normalizeIds(value.favoriteIds, 100),
      recentIds: normalizeIds(value.recentIds, 10),
      fontSize: FONT_SIZES.has(value.fontSize) ? value.fontSize : DEFAULTS.fontSize,
    };
  }

  function load(storage) {
    if (!storage) return normalize(null);
    try {
      const raw = storage.getItem(STORAGE_KEY);
      return raw ? normalize(JSON.parse(raw)) : normalize(null);
    } catch {
      return normalize(null);
    }
  }

  function save(storage, value) {
    if (!storage) return false;
    try {
      const normalized = normalize({ version: 1, ...value });
      storage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ...normalized }));
      return true;
    } catch {
      return false;
    }
  }

  function clear(storage) {
    if (!storage) return false;
    try {
      storage.removeItem(STORAGE_KEY);
      return true;
    } catch {
      return false;
    }
  }

  window.GeaEducationPreferences = Object.freeze({ load, save, clear });
})();
