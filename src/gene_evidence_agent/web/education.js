(() => {
  const byId = (id) => document.getElementById(id);
  let preferenceStorage = null;
  try { preferenceStorage = window.localStorage; } catch { /* browser storage may be unavailable */ }
  const preferenceStore = window.GeaEducationPreferences;
  const savedPreferences = preferenceStore.load(preferenceStorage);
  const learning = window.GeaEducationLearning;
  const learningData = learning.load(preferenceStorage);
  const state = {
    kind: "cancer", query: "", category: "", entries: [], selected: null, scopeNotice: null,
    favoriteIds: new Set(savedPreferences.favoriteIds), recentIds: [...savedPreferences.recentIds],
    favoritesOnly: false, recentOnly: false, readerSize: savedPreferences.fontSize,
    sessionId: null, pendingSubmit: null, pendingJob: null, modelMode: "unknown", polling: false, sending: false,
    contextEntries: [], importing: false,
    reading: null, allEntries: [], entryEpoch: 0, lastQuestion: "", catalogPromise: null,
  };
  const sourceHosts = new Set(["cancer.gov", "www.cancer.gov", "cancer.org", "www.cancer.org", "ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "medlineplus.gov", "www.genome.gov", "who.int", "www.who.int", "platform.opentargets.org", "cancerresearchuk.org", "www.cancerresearchuk.org", "www.iarc.who.int"]);
  const mediaImageHosts = new Set(["cancer.gov", "www.cancer.gov"]);
  const attachments = window.GeaAttachments?.create({
    input: "education-chat-input", channel: "education",
    canUpload: () => !!state.selected && !state.pendingJob && !state.pendingSubmit && !state.sending,
    getSession: async () => {
      if (!state.sessionId) {
        const session = await api("/api/education/chat/session", { entry_id: state.selected.id });
        state.sessionId = session.session_id;
      }
      return state.sessionId;
    },
    onBusy: (busy) => { state.uploading = busy; setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready"); }
  });
  const errorText = {
    education_context_invalid: "知识卡片选择无效，请重新加入。",
    education_context_limit: "一次最多组合 6 张知识卡片，请移除部分卡片后再加入。",
    invalid_model_output: "回答格式不完整，未接受该回答。可重新表述问题后主动发送；不会自动重试。",
    invalid_citation: "回答引用未通过校验，未接受该回答。原始词条和先前对话仍保留。",
    session_not_found: "会话已过期；请新建科普对话后继续。",
    job_not_found: "原任务已过期；可新建科普对话。不会重新提交旧问题。",
    attachment_not_found: "参考文件已过期，请新建对话并重新添加文件。",
    context_too_large: "文件与对话内容超过单次输入上限；请新建对话并缩短文件。此次未调用模型。",
    personal_medical_advice_or_sensitive_data: "这类个人诊疗问题或敏感信息不会发送给模型。请移除个人信息；个体诊断、治疗和用药请咨询合格医疗专业人员。",
    education_topic_changed: "当前会话绑定了另一科普主题，请新建对话后再提问。",
    education_entry_not_found: "未找到该科普条目，请刷新目录。",
    education_source_unavailable: "当前条目资料不可用，未发送模型请求。",
    model_disabled: "科普对话模型未在本机服务端启用；你仍可阅读词条和来源。",
    model_budget: "科普对话预算暂不可用或已达到限制；原始科普条目仍可阅读。",
    timeout: "模型请求超时；本次不会自动重试。",
    rate_limited: "模型服务当前限流；本次不会自动重试。",
    empty_content: "模型返回了空内容；本次不会自动重试。",
    output_truncated: "模型回答被截断；本次不会自动重试。",
    unknown_usage: "本次用量无法确认；后续模型调用受账本策略控制。",
  };

  async function api(path, payload) {
    const response = await fetch(path, payload ? {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), cache: "no-store",
    } : { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(errorText[data.error] || "科普服务暂不可用，请稍后检查本机服务状态。");
      error.code = data.error;
      throw error;
    }
    return data;
  }

  function safeSourceUrl(value) {
    try {
      const url = new URL(value);
      return url.protocol === "https:" && sourceHosts.has(url.hostname) ? url.href : null;
    } catch { return null; }
  }

  function safeMediaAssetUrl(media) {
    if (media.display_mode === "local_asset" && typeof media.asset_url === "string"
      && media.asset_url.startsWith("/") && !media.asset_url.startsWith("//")) return media.asset_url;
    try {
      const url = new URL(media.asset_url);
      return media.display_mode === "approved_embed" && url.protocol === "https:"
        && mediaImageHosts.has(url.hostname) ? url.href : null;
    } catch { return null; }
  }

  function safeVideoSourceUrl(value) {
    const sourceUrl = safeSourceUrl(value);
    if (!sourceUrl) return null;
    try {
      const url = new URL(sourceUrl);
      return !url.username && !url.password && (!url.port || url.port === "443") ? url.href : null;
    } catch { return null; }
  }

  function safeRegisteredSourceUrl(value) {
    const sourceUrl = safeSourceUrl(value);
    if (!sourceUrl) return null;
    try {
      const url = new URL(sourceUrl);
      return !url.username && !url.password && (!url.port || url.port === "443") ? url.href : null;
    } catch { return null; }
  }

  function renderEducationMedia(mediaItems) {
    const visible = (mediaItems || []).filter((item) => ["image", "video"].includes(item.kind)
      && item.publication_status === "published"
      && item.license_status === "rights_checked"
      && (item.kind === "video" ? safeVideoSourceUrl(item.source_page_url) : safeSourceUrl(item.source_page_url)));
    if (!visible.length) return null;
    const hasImages = visible.some((item) => item.kind === "image");
    const hasVideos = visible.some((item) => item.kind === "video");
    const sectionLabel = hasImages && hasVideos ? "专业图示与视频" : hasVideos ? "专业视频" : "专业图示";
    const section = document.createElement("section");
    section.className = "education-media-section";
    section.setAttribute("aria-label", sectionLabel);
    const heading = document.createElement("h3");
    heading.className = "education-media-heading";
    heading.textContent = sectionLabel;
    section.append(heading);
    visible.forEach((media) => {
      const sourceUrl = media.kind === "video"
        ? safeVideoSourceUrl(media.source_page_url)
        : safeSourceUrl(media.source_page_url);
      if (!sourceUrl) return;
      if (media.kind === "video") {
        const card = document.createElement("article");
        card.className = "education-video-source-card";
        const badge = document.createElement("span");
        badge.className = "education-video-source-badge";
        badge.textContent = "专业视频";
        const title = document.createElement("p");
        title.className = "education-media-title";
        title.textContent = media.title;
        const description = document.createElement("p");
        description.className = "education-video-source-description";
        description.textContent = media.caption;
        const rawLanguage = typeof media.language === "string" ? media.language.trim() : "";
        const language = rawLanguage && !["unknown", "未核实"].includes(rawLanguage.toLowerCase())
          ? rawLanguage : "未核实";
        const durationSeconds = Number.isInteger(media.duration_seconds) && media.duration_seconds >= 0
          ? media.duration_seconds : null;
        const duration = durationSeconds === null
          ? "未核实"
          : `${Math.floor(durationSeconds / 60)}:${String(durationSeconds % 60).padStart(2, "0")}`;
        const metadata = document.createElement("p");
        metadata.className = "education-video-source-meta";
        metadata.textContent = `语言：${language}　时长：${duration}`;
        const note = document.createElement("p");
        note.className = "education-video-source-note";
        note.textContent = "本页仅链接原始来源。视频版本、语言、字幕/文字稿和播放状态可能因版本或原站变化而异，请以原站当前播放器为准。";
        const publisher = document.createElement("span");
        publisher.className = "education-video-source-publisher";
        publisher.textContent = `来源：${media.publisher}`;
        const source = document.createElement("a");
        source.href = sourceUrl;
        source.target = "_blank";
        source.rel = "noopener noreferrer";
        source.referrerPolicy = "no-referrer";
        source.setAttribute("aria-label", `打开 ${media.publisher} 来源网站查看视频（新标签页）`);
        source.textContent = "打开来源网站查看视频";
        card.append(badge, title, description, metadata, note, publisher, source);
        section.append(card);
        return;
      }
      const figure = document.createElement("figure");
      figure.className = "education-media-figure";
      if (media.kind === "image") {
        const assetUrl = safeMediaAssetUrl(media);
        if (assetUrl) {
          const image = document.createElement("img");
          image.className = "education-media-image";
          image.src = assetUrl;
          image.alt = media.alt_text;
          image.loading = "lazy";
          image.decoding = "async";
          image.referrerPolicy = "no-referrer";
          let imageControl = image;
          const dialog = document.createElement("dialog");
          if (typeof dialog.showModal === "function") {
            dialog.className = "education-media-dialog";
            const dialogTitle = document.createElement("h3");
            dialogTitle.className = "education-media-dialog-title";
            dialogTitle.id = `education-media-dialog-title-${media.id}`;
            dialogTitle.textContent = media.title;
            dialog.setAttribute("aria-labelledby", dialogTitle.id);
            const close = document.createElement("button");
            close.type = "button";
            close.className = "education-media-dialog-close";
            close.setAttribute("aria-label", "关闭大图");
            close.textContent = "关闭";
            close.addEventListener("click", () => dialog.close());
            const expandedImage = document.createElement("img");
            expandedImage.className = "education-media-dialog-image";
            expandedImage.src = assetUrl;
            expandedImage.alt = media.alt_text;
            expandedImage.decoding = "async";
            expandedImage.referrerPolicy = "no-referrer";
            const expandedCaption = document.createElement("p");
            expandedCaption.className = "education-media-dialog-caption";
            expandedCaption.textContent = media.caption;
            const original = document.createElement("a");
            original.href = sourceUrl;
            original.target = "_blank";
            original.rel = "noopener noreferrer";
            original.referrerPolicy = "no-referrer";
            original.setAttribute("aria-label", `查看 ${media.publisher} 原始页面：${media.source_page_title}`);
            original.textContent = media.source_page_title;
            dialog.append(close, dialogTitle, expandedImage, expandedCaption, original);
            dialog.addEventListener("click", (event) => {
              if (event.target === dialog) dialog.close();
            });
            const enlarge = document.createElement("button");
            enlarge.type = "button";
            enlarge.className = "education-media-enlarge";
            enlarge.setAttribute("aria-label", `放大查看：${media.title}`);
            enlarge.append(image);
            enlarge.addEventListener("click", () => dialog.showModal());
            imageControl = enlarge;
            figure.append(enlarge, dialog);
          } else {
            figure.append(image);
          }
          const fallback = document.createElement("p");
          fallback.className = "education-media-fallback hidden";
          fallback.textContent = `图片暂时无法载入；可打开 ${media.publisher} 原始页面查看。`;
          image.addEventListener("error", () => {
            imageControl.classList.add("hidden");
            fallback.classList.remove("hidden");
          }, { once: true });
          figure.append(fallback);
        }
      }
      const caption = document.createElement("figcaption");
      caption.className = "education-media-caption";
      const title = document.createElement("p");
      title.className = "education-media-title";
      title.textContent = media.title;
      const description = document.createElement("p");
      description.textContent = `${media.caption}${media.kind === "image" ? "（点击图片可放大查看）" : ""}`;
      const publisher = document.createElement("span");
      publisher.textContent = `来源：${media.publisher}`;
      const source = document.createElement("a");
      source.href = sourceUrl;
      source.target = "_blank";
      source.rel = "noopener noreferrer";
      source.referrerPolicy = "no-referrer";
      source.setAttribute("aria-label", `打开原始产品页面：${media.source_page_title}`);
      source.textContent = media.source_page_title;
      caption.append(title, description, publisher, source);
      figure.append(caption);
      section.append(figure);
    });
    return section.childNodes.length > 1 ? section : null;
  }

  function renderEducationFaqs(faqs, sources) {
    const sourceById = new Map((Array.isArray(sources) ? sources : [])
      .filter((source) => source && source.review_status === "source_checked"
        && safeRegisteredSourceUrl(source.url))
      .map((source) => [source.id, source]));
    const visible = (Array.isArray(faqs) ? faqs : []).slice(0, 4).map((faq) => ({
      faq,
      sources: [...new Set(Array.isArray(faq.source_ids) ? faq.source_ids : [])]
        .map((sourceId) => sourceById.get(sourceId))
        .filter(Boolean),
    })).filter(({ faq, sources: faqSources }) =>
      typeof faq.question === "string" && faq.question.trim()
      && typeof faq.answer === "string" && faq.answer.trim()
      && faqSources.length > 0
    );
    if (!visible.length) return null;

    const section = document.createElement("section");
    section.className = "education-faq-section";
    section.setAttribute("aria-label", "常见问题");
    const heading = document.createElement("h3");
    heading.className = "education-faq-heading";
    heading.textContent = "常见问题";
    section.append(heading);

    visible.forEach(({ faq, sources: faqSources }) => {
      const details = document.createElement("details");
      details.className = "education-faq-item";
      const question = document.createElement("summary");
      question.textContent = faq.question.trim();
      const answer = document.createElement("p");
      answer.className = "education-faq-answer";
      answer.textContent = faq.answer.trim();
      const references = document.createElement("div");
      references.className = "education-faq-sources";
      references.setAttribute("aria-label", "参考来源");
      faqSources.forEach((source) => {
        const href = safeRegisteredSourceUrl(source.url);
        if (!href) return;
        const link = document.createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.referrerPolicy = "no-referrer";
        link.setAttribute("aria-label", `打开 ${source.publisher} 来源：${source.title}（新标签页）`);
        link.textContent = `来源：${source.publisher} — ${source.title}`;
        references.append(link);
      });
      details.append(question, answer, references);
      section.append(details);
    });
    return section;
  }

  function updateNavigation(educationOn, { focusSearch = true } = {}) {
    if (!educationOn) state.paneFocusEpoch = (state.paneFocusEpoch || 0) + 1;
    const panel = byId("education-view");
    const toggle = byId("education-toggle");
    const researchToggle = byId("research-toggle");
    document.querySelector(".workspace-content").classList.toggle("hidden", educationOn);
    byId("chat-panel").classList.toggle("hidden", educationOn);
    byId("chat-backdrop").classList.toggle("hidden", educationOn);
    byId("history-drawer").classList.add("hidden");
    byId("drawer-backdrop").classList.add("hidden");
    document.body.classList.remove("chat-open");
    panel.classList.toggle("hidden", !educationOn);
    toggle.setAttribute("aria-pressed", String(educationOn));
    toggle.classList.toggle("is-current", educationOn);
    researchToggle.classList.toggle("is-current", !educationOn);
    if (educationOn) {
      researchToggle.removeAttribute("aria-current");
      toggle.setAttribute("aria-current", "page");
    } else {
      researchToggle.setAttribute("aria-current", "page");
      toggle.removeAttribute("aria-current");
    }
    byId("chat-toggle").classList.toggle("hidden", educationOn);
    byId("history-toggle").classList.toggle("hidden", educationOn);
    if (educationOn) {
      document.body.classList.add("education-mode");
      if (focusSearch) setEducationPane(byId("education-grid").dataset.activePane || "catalog", true);
    } else {
      document.body.classList.remove("education-mode");
      document.dispatchEvent(new CustomEvent("gea:education-view-hidden"));
      researchToggle.focus({ preventScroll: true });
    }
  }

  function setEducationPane(pane, focus = false) {
    const focusEpoch = (state.paneFocusEpoch || 0) + 1;
    state.paneFocusEpoch = focusEpoch;
    const grid = byId("education-grid");
    grid.dataset.activePane = pane;
    byId("education-view").dataset.learning = String(pane !== "catalog");
    if (pane === "catalog" && focus) byId("education-view").dataset.catalogOpen = "true";
    document.querySelectorAll(".education-mobile-nav [data-education-pane]").forEach((button) => {
      const selected = button.dataset.educationPane === pane;
      button.setAttribute("aria-pressed", String(selected));
      button.disabled = !state.selected && button.dataset.educationPane !== "catalog";
    });
    if (focus) {
      const target = pane === "catalog"
        ? byId("education-search")
        : pane === "chat"
          ? byId("education-chat-input")
          : byId(`education-${pane}`);
      requestAnimationFrame(() => {
        if (state.paneFocusEpoch !== focusEpoch || grid.dataset.activePane !== pane) return;
        target.scrollIntoView({ block: pane === "chat" ? "center" : "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        target.focus({ preventScroll: true });
      });
    }
  }

  function currentEntries() {
    return state.entries.filter((entry) =>
      (!state.category || entry.category === state.category)
      && (!state.favoritesOnly || state.favoriteIds.has(entry.id))
      && (!state.recentOnly || state.recentIds.includes(entry.id))
    );
  }

  function persistPreferences() {
    const saved = preferenceStore.save(preferenceStorage, {
      favoriteIds: [...state.favoriteIds], recentIds: state.recentIds, fontSize: state.readerSize,
    });
    byId("education-preferences-status").textContent = saved
      ? "收藏与阅读设置仅存本机。"
      : "浏览器未允许本机保存；当前页面内仍可使用这些设置。";
    return saved;
  }

  function renderPreferenceControls() {
    const favoritesButton = byId("education-favorites-toggle");
    favoritesButton.setAttribute("aria-pressed", String(state.favoritesOnly));
    byId("education-favorites-count").textContent = String(state.favoriteIds.size);
    byId("education-recent-toggle").setAttribute("aria-pressed", String(state.recentOnly));
    byId("education-recent-count").textContent = String(state.recentIds.length);
    byId("education-preferences-reset").classList.toggle(
      "hidden", state.favoriteIds.size === 0 && state.recentIds.length === 0 && state.readerSize === "standard",
    );
    const detail = byId("education-detail");
    detail.dataset.readerSize = state.readerSize;
    detail.querySelectorAll("[data-education-font-size]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.educationFontSize === state.readerSize));
    });
    const favoriteButton = byId("education-entry-favorite");
    if (favoriteButton && state.reading) {
      const isFavorite = state.favoriteIds.has(state.reading.id);
      favoriteButton.setAttribute("aria-pressed", String(isFavorite));
      favoriteButton.textContent = isFavorite ? "已收藏" : "收藏主题";
    }
  }

  function toggleFavorite(entry) {
    if (state.favoriteIds.has(entry.id)) state.favoriteIds.delete(entry.id);
    else state.favoriteIds.add(entry.id);
    persistPreferences();
    renderPreferenceControls();
    renderList();
  }

  function changeReaderSize(size) {
    if (!new Set(["standard", "large", "largest"]).has(size)) return;
    state.readerSize = size;
    persistPreferences();
    renderPreferenceControls();
  }

  function renderCategories() {
    const select = byId("education-category");
    const prior = state.category;
    const categories = [...new Set(state.entries.map((entry) => entry.category))].sort((a, b) => a.localeCompare(b, "zh-CN"));
    select.replaceChildren(new Option("全部主题", ""));
    categories.forEach((category) => select.add(new Option(category, category)));
    select.value = categories.includes(prior) ? prior : "";
    state.category = select.value;
    renderCategoryFilters();
  }

  function renderCategoryFilters() {
    const cancer = state.kind === "cancer";
    byId("education-catalog").dataset.kind = state.kind;
    byId("education-catalog-title").textContent = cancer ? "按分类，认识癌症" : state.kind === "gene" ? "认识一个基因" : "读懂医学词语";
    byId("education-catalog-note").textContent = cancer ? "入门目录，非完整病理分类。" : "选择感兴趣的主题。";
    const filters = byId("education-category-filters");
    filters.replaceChildren();
    filters.classList.toggle("hidden", !cancer);
    if (!cancer) return;
    const counts = new Map();
    state.entries.forEach(entry => counts.set(entry.category, (counts.get(entry.category) || 0) + 1));
    const categories = [["", state.entries.length], ...counts];
    for (const [category, count] of categories) {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(state.category === category));
      const label = document.createElement("span");
      label.textContent = category || "全部主题";
      const number = document.createElement("span");
      number.className = "education-category-count";
      number.textContent = String(count);
      button.append(label, number);
      button.addEventListener("click", () => {
        state.category = category;
        byId("education-category").value = category;
        renderCategoryFilters();
        renderList();
        filters.querySelector('[aria-pressed="true"]')?.focus({ preventScroll: true });
      });
      filters.append(button);
    }
  }

  function renderList() {
    const list = byId("education-entry-list");
    const visible = currentEntries();
    byId("education-topic-count").textContent = `${visible.length} 个主题`;
    byId("education-filter-reset").classList.toggle("hidden", !state.query && !state.category && !state.favoritesOnly && !state.recentOnly);
    list.replaceChildren();
    if (!visible.length) {
      const empty = document.createElement("p");
      empty.className = "education-placeholder";
      empty.textContent = state.favoritesOnly && state.favoriteIds.size === 0
        ? "你还没有收藏主题。打开一个词条，点“收藏主题”，就能在这里快速找到它。"
        : state.recentOnly && state.recentIds.length === 0
          ? "这里还没有阅读记录。选择一个科普主题后，它会出现在这里。"
          : state.favoritesOnly
            ? "当前搜索或类别下没有已收藏的主题。可以清除搜索词、切换类别或关闭“只看收藏”。"
            : state.recentOnly
              ? "当前搜索或类别下没有最近阅读的主题。可以清除搜索词、切换类别或关闭“最近阅读”。"
              : "没有匹配的科普词条。可调整类别或搜索词。";
      list.append(empty);
      return;
    }
    const groups = new Map();
    visible.forEach((entry) => {
      if (!groups.has(entry.category)) groups.set(entry.category, []);
      groups.get(entry.category).push(entry);
    });
    for (const [category, entries] of groups) {
      const section = document.createElement("section");
      section.className = "education-list-group";
      const heading = document.createElement("h3");
      heading.textContent = category;
      const count = document.createElement("span");
      count.className = "education-group-count";
      count.textContent = String(entries.length);
      heading.append(count);
      section.append(heading);
      for (const entry of entries) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "education-entry-button";
        button.setAttribute("role", "listitem");
        button.setAttribute("aria-pressed", String(state.reading?.id === entry.id));
        button.disabled = !!state.pendingJob || !!state.pendingSubmit || state.sending || state.uploading || state.importing;
        const title = document.createElement("strong");
        title.textContent = entry.name;
        const description = document.createElement("span");
        description.textContent = entry.summary;
        button.append(title, description);
        if (entry.kind === "cancer" && entry.name.includes("分类入口")) {
          const scope = document.createElement("span");
          scope.className = "education-topic-kind";
          scope.textContent = "分类导览";
          button.append(scope);
        }
        if (state.favoriteIds.has(entry.id)) {
          const mark = document.createElement("span");
          mark.className = "education-entry-favorite-mark";
          mark.textContent = "已收藏";
          button.append(mark);
        }
        button.addEventListener("click", () => void selectEntry(entry.id));
        const row = document.createElement("div"); row.className = "education-topic-row";
        const importButton = document.createElement("button");
        importButton.type = "button"; importButton.className = "education-card-import";
        importButton.dataset.educationImport = entry.id;
        const imported = state.contextEntries.some((item) => item.id === entry.id);
        importButton.textContent = imported ? "已加入" : "结合提问";
        importButton.setAttribute("aria-label", `${imported ? "查看已加入的" : "将"}${entry.name}${imported ? "对话上下文" : "卡片加入对话"}`);
        importButton.setAttribute("aria-pressed", String(imported));
        importButton.disabled = !!state.pendingJob || !!state.pendingSubmit || state.sending || state.uploading || state.importing;
        importButton.addEventListener("click", () => void importKnowledgeCard(entry.id));
        row.append(button, importButton); section.append(row);
      }
      list.append(section);
    }
  }

  function renderSearchScopeNotice(notice) {
    const container = byId("education-search-scope-notice");
    container.replaceChildren();
    container.classList.toggle("hidden", !notice);
    if (!notice) return;
    const title = document.createElement("strong");
    title.textContent = notice.title;
    const messageText = document.createElement("p");
    messageText.textContent = notice.message;
    const sources = document.createElement("div");
    sources.className = "education-search-scope-sources";
    (notice.sources || []).forEach((source) => {
      const href = safeSourceUrl(source.url);
      if (!href) return;
      const link = document.createElement("a");
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title;
      sources.append(link);
    });
    container.append(title, messageText);
    if (sources.childNodes.length) container.append(sources);
  }

  function renderEntry(entry) {
    const panel = byId("education-detail");
    panel.replaceChildren();
    panel.dataset.readerSize = state.readerSize;
    const kicker = document.createElement("div");
    kicker.className = "card-kicker";
    kicker.textContent = ({ cancer: "认识癌症", gene: "了解基因", concept: "读懂一个词" })[entry.kind] || "科普学习";
    const tools = document.createElement("div");
    tools.className = "education-reader-tools";
    const fontControls = document.createElement("div");
    fontControls.className = "education-reader-size";
    fontControls.setAttribute("role", "group");
    fontControls.setAttribute("aria-label", "阅读文字大小");
    const sizeLabel = document.createElement("span");
    sizeLabel.textContent = "字号";
    fontControls.append(sizeLabel);
    [["standard", "标准"], ["large", "较大"], ["largest", "最大"]].forEach(([size, label]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.educationFontSize = size;
      button.setAttribute("aria-label", `${label}字号`);
      button.setAttribute("aria-pressed", String(size === state.readerSize));
      button.textContent = label;
      button.addEventListener("click", () => changeReaderSize(size));
      fontControls.append(button);
    });
    const favorite = document.createElement("button");
    favorite.type = "button";
    favorite.id = "education-entry-favorite";
    favorite.className = "education-favorite-button";
    favorite.setAttribute("aria-pressed", String(state.favoriteIds.has(entry.id)));
    favorite.textContent = state.favoriteIds.has(entry.id) ? "已收藏" : "收藏主题";
    favorite.addEventListener("click", () => toggleFavorite(entry));
    tools.append(fontControls, favorite);
    const title = document.createElement("h2");
    title.className = "education-entry-title";
    title.textContent = entry.name;
    const category = document.createElement("span");
    category.className = "education-category-badge";
    category.textContent = entry.category;
    const summary = document.createElement("p");
    summary.className = "education-entry-summary";
    summary.textContent = entry.summary;
    const body = document.createElement("p");
    body.className = "education-entry-body";
    body.textContent = entry.body;
    const faqs = renderEducationFaqs(entry.faqs, entry.faq_sources);
    const media = renderEducationMedia(entry.media);
    const aliases = document.createElement("p");
    aliases.className = "education-aliases";
    aliases.textContent = `相关名称：${(entry.aliases || []).join(" · ")}`;
    const provenance = document.createElement("div");
    provenance.className = "education-provenance";
    const provenanceLabel = document.createElement("span");
    provenanceLabel.textContent = `来源核对：${entry.checked_at} · 状态：${entry.review_status === "source_checked" ? "已核对来源，未独立专家审核" : "草稿"}`;
    provenance.append(provenanceLabel);
    const href = safeSourceUrl(entry.source_url);
    if (href) {
      const link = document.createElement("a");
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `打开原始来源：${entry.source_title} ↗`;
      provenance.append(link);
    }
    const section = (label, content, id) => {
      const details = document.createElement("details");
      details.className = "education-reading-section";
      details.id = id;
      const heading = document.createElement("summary");
      heading.textContent = label;
      details.append(heading, content);
      return details;
    };
    const explanation = section("继续了解", body, "education-explanation");
    const next = document.createElement("div");
    next.className = "education-reader-next";
    const readMore = document.createElement("button");
    readMore.type = "button";
    readMore.className = "primary-button";
    readMore.textContent = "继续了解";
    readMore.addEventListener("click", () => {
      explanation.open = true;
      explanation.querySelector("summary").focus({ preventScroll: true });
      explanation.scrollIntoView({ block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    });
    const ask = document.createElement("button");
    ask.type = "button";
    ask.className = "quiet-button";
    ask.id = "education-import-current";
    ask.dataset.educationImport = entry.id;
    ask.textContent = "结合这张卡提问";
    ask.addEventListener("click", () => void importKnowledgeCard(entry.id));
    next.append(readMore, ask);
    if (state.selected && state.selected.id !== entry.id) {
      const switchTopic = document.createElement("button"); switchTopic.type = "button";
      switchTopic.id = "education-switch-topic"; switchTopic.className = "back-link";
      switchTopic.textContent = "以此主题新建对话";
      switchTopic.addEventListener("click", () => {
        if (!window.confirm("新建此主题的对话？当前对话将结束，未保存问答、草稿和文件选择会清空。也可以取消并选择结合卡片提问。")) return;
        void selectEntry(entry.id, { switchTopic: true });
      });
      next.append(switchTopic);
    }
    const safety = document.createElement("p");
    safety.className = "education-safety";
    safety.textContent = "一般知识学习，不作个人诊疗判断。";
    panel.append(kicker, title, category, summary, next, explanation);
    if (faqs) panel.append(section("常见疑问", faqs, "education-faq-details"));
    if (media) panel.append(section("图片与视频", media, "education-media-details"));
    const sourceContent = document.createElement("div");
    sourceContent.append(aliases, provenance);
    panel.append(section("资料来源与说明", sourceContent, "education-source-details"), tools, safety);
    renderLearningAids(entry);
    panel.querySelectorAll(".education-reading-section").forEach(details => {
      details.addEventListener("toggle", () => { if (details.open) rememberReading(entry.id, details.id); });
    });
  }

  function renderEmptyDetail() {
    const panel = byId("education-detail");
    panel.replaceChildren();
    const stateCard = document.createElement("div");
    stateCard.className = "education-placeholder-state";
    const icon = document.createElement("span");
    icon.className = "education-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "✳";
    const title = document.createElement("h2");
    title.textContent = "从一个主题开始";
    const text = document.createElement("p");
    text.textContent = "选一个主题，一次学一点。";
    const note = document.createElement("p");
    note.className = "education-review-note";
    note.textContent = "资料已做来源核对；这不等于独立医学专家审核。";
    stateCard.append(icon, title, text, note);
    panel.append(stateCard);
  }

  function renderCardContext() {
    const box = byId("education-chat-context"), list = byId("education-chat-context-list");
    list.replaceChildren();
    box.classList.toggle("hidden", !state.contextEntries.length);
    for (const entry of state.contextEntries) {
      const chip = document.createElement("div"); chip.className = "education-context-chip";
      chip.setAttribute("role", "listitem");
      const label = document.createElement("span");
      label.textContent = entry.name + (entry.id === state.selected?.id ? " · 当前主题" : "");
      chip.append(label);
      if (entry.id !== state.selected?.id) {
        const remove = document.createElement("button"); remove.type = "button";
        remove.textContent = "×"; remove.setAttribute("aria-label", `移除${entry.name}卡片`);
        remove.disabled = !!state.pendingJob || !!state.pendingSubmit || state.sending || state.uploading || state.importing;
        remove.addEventListener("click", () => {
          if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading || state.importing) return;
          state.contextEntries = state.contextEntries.filter((item) => item.id !== entry.id);
          renderCardContext(); renderList();
          byId("education-chat-status").textContent = `已从后续提问上下文移除「${entry.name}」。`;
        });
        chip.append(remove);
      }
      list.append(chip);
    }
  }

  async function importKnowledgeCard(entryId) {
    if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading || state.importing) return false;
    const draft = byId("education-chat-input").value;
    state.importing = true; setChatEnabled(false);
    let success = false;
    try {
      if (!state.selected) await selectEntry(entryId);
      const entry = state.selected?.id === entryId ? state.selected : await api(`/api/education/entry?id=${encodeURIComponent(entryId)}`);
      if (!state.selected || entry.review_status !== "source_checked") throw new Error("当前卡片资料不可用。");
      const entries = [...state.contextEntries];
      if (!entries.some((item) => item.id === state.selected.id)) entries.unshift(state.selected);
      if (!entries.some((item) => item.id === entry.id)) entries.push(entry);
      if (entries.length > 6) throw new Error(errorText.education_context_limit);
      state.contextEntries = entries;
      byId("education-chat-input").value = draft;
      byId("education-chat-status").textContent = `已加入「${entry.name}」的简介、正文和常见疑问。写下问题后发送即可；尚未调用模型。`;
      success = true;
    } catch (error) { byId("education-chat-status").textContent = error.message; }
    finally {
      state.importing = false; renderCardContext(); renderList();
      setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
    }
    if (success) { setEducationPane("chat", true); byId("education-chat-input").focus(); }
    return success;
  }

  function message(role, text, citations = [], mock = false) {
    const container = document.createElement("article");
    container.className = `education-message education-message-${role}`;
    const label = document.createElement("strong");
    label.textContent = role === "user" ? "你" : "科普助手";
    const content = document.createElement("p");
    const abbreviated = role === "assistant" && text.length > 260;
    content.textContent = abbreviated ? text.slice(0, 240) + "…" : text;
    container.append(label, content);
    if (abbreviated) {
      const details = document.createElement("details"); details.className = "education-answer-full";
      const summary = document.createElement("summary"); summary.textContent = "展开完整回答";
      const full = document.createElement("p"); full.textContent = text;
      details.append(summary, full); container.append(details);
      details.addEventListener("toggle", () => { content.hidden = details.open; });
    }
    if (citations.length) {
      const links = document.createElement("div");
      links.className = "education-chat-citations";
      citations.forEach((citation) => {
        const href = safeSourceUrl(citation.url);
        if (!href) return;
        const link = document.createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = `${citation.source_id} · ${citation.title} ↗`;
        links.append(link);
      });
      if (links.childNodes.length) container.append(links);
    }
    if (role === "user") state.lastQuestion = text;
    if (role === "assistant") {
      state.lastAnswer = text;
      const question = state.lastQuestion || "围绕当前主题的提问";
      const save = document.createElement("button"); save.type = "button"; save.className = "back-link education-save-answer";
      save.textContent = "保存这条问答";
      save.addEventListener("click", () => { if (saveAnswer(question, text, citations, mock)) { save.textContent = "已保存到本机"; save.disabled = true; } });
      container.append(save);
      renderSuggestedQuestions();
      const terms = learning.glossary(text, state.allEntries, state.selected?.id);
      if (terms.length) {
        const glossary = document.createElement("div"); glossary.className = "education-term-links";
        terms.forEach(term => { const button = document.createElement("button"); button.type = "button";
          button.textContent = `解释「${term.name}」`; button.addEventListener("click", () => openTerm(term)); glossary.append(button); });
        container.append(glossary);
      }
    }
    byId("education-chat-messages").append(container);
    byId("education-chat-messages").scrollTop = byId("education-chat-messages").scrollHeight;
  }

  function setChatEnabled(enabled) {
    enabled = enabled && !state.uploading && !state.importing;
    const busy = !!state.pendingJob || !!state.pendingSubmit || state.sending || state.uploading || state.importing;
    const input = byId("education-chat-input");
    input.disabled = !enabled || !!state.pendingJob || !!state.pendingSubmit || state.sending;
    byId("education-chat-send").disabled = !enabled || !!state.pendingJob || !!state.pendingSubmit || state.sending;
    byId("education-chat-clear").disabled = !state.sessionId || busy;
    document.querySelectorAll(".education-chat-prompt").forEach((button) => {
      button.disabled = !enabled || !state.selected || busy;
    });
    document.querySelectorAll(".education-entry-button").forEach((button) => { button.disabled = !!busy; });
    document.querySelectorAll("[data-education-kind]").forEach((button) => { button.disabled = !!busy; });
    document.querySelectorAll("[data-learning-kind]").forEach((button) => { button.disabled = !!busy; });
    document.querySelectorAll("[data-education-import], .education-context-chip button, .education-suggested-question, #education-switch-topic, #education-home-question-form button").forEach((button) => { button.disabled = !!busy; });
  }

  function educationPromptText(kind, topicName) {
    const topic = String(topicName || "当前主题").trim() || "当前主题";
    const prompts = {
      simple: `请用更简单的话解释“${topic}”，少用专业术语。`,
      beginner: `我不知道从哪里开始。请给我 3 个适合初学者的问题，帮助我了解“${topic}”。`,
      term: `请用白话解释“${topic}”中最重要的一个专业词。`,
    };
    return prompts[kind] || "";
  }

  function applyEducationPrompt(kind) {
    if (!state.selected || state.pendingJob || state.pendingSubmit || state.sending
      || !["mock", "ready"].includes(state.modelMode)) return false;
    const input = byId("education-chat-input");
    if (input.disabled) return false;
    const prompt = educationPromptText(kind, state.selected.name);
    if (!prompt) return false;
    input.value = prompt;
    input.focus();
    byId("education-chat-status").textContent = "问题已填好，还没有发送。点击“发送问题”后才会提交。";
    return true;
  }

  async function loadCatalog() {
    const epoch = (state.catalogEpoch || 0) + 1;
    state.catalogEpoch = epoch;
    try {
      const params = new URLSearchParams({ kind: state.kind });
      if (state.query) params.set("q", state.query);
      const response = await api(`/api/education/catalog?${params}`);
      if (state.catalogEpoch !== epoch) return;
      state.entries = response.entries;
      state.scopeNotice = response.scope_notice || null;
      renderCategories();
      renderSearchScopeNotice(state.scopeNotice);
      renderList();
    } catch {
      if (state.catalogEpoch !== epoch) return;
      const list = byId("education-entry-list");
      list.replaceChildren();
      const error = document.createElement("p");
      error.className = "education-placeholder education-error";
      error.textContent = "无法加载科普目录。请确认 Gene Evidence 本机服务仍在运行。";
      renderSearchScopeNotice(null);
      list.append(error);
    }
  }

  async function ensureLearningCatalog() {
    if (!state.catalogPromise) state.catalogPromise = api("/api/education/catalog").then(data => {
      state.allEntries = data.entries.filter(entry => entry.review_status === "source_checked");
      refreshContinueButton();
      return state.allEntries;
    }).catch(error => { state.catalogPromise = null; throw error; });
    return state.catalogPromise;
  }

  function rememberReading(entryId, section = "summary") {
    learningData.progress = { entryId, section };
    learning.save(preferenceStorage, learningData);
    refreshContinueButton();
  }

  function refreshContinueButton() {
    const entry = state.allEntries.find(item => item.id === (learningData.progress?.entryId || state.recentIds[0]));
    const button = byId("education-continue-learning");
    button.classList.toggle("hidden", !entry);
    if (entry) { button.textContent = `继续阅读：${entry.name}`; button.dataset.entryId = entry.id; }
  }

  function fillQuestion(question) {
    if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading || state.importing) return;
    const input = byId("education-chat-input");
    if (input.value.trim() && input.value.trim() !== question && !window.confirm("替换当前未发送的问题？")) return;
    input.value = question;
    setEducationPane("chat", true);
    byId("education-chat-status").textContent = "问题已填入；点击发送问题后才会调用模型。";
  }

  function renderSuggestedQuestions() {
    let host = byId("education-next-questions");
    if (!host) {
      host = document.createElement("details"); host.id = "education-next-questions";
      host.className = "education-next-questions"; host.setAttribute("role", "group");
      host.setAttribute("aria-label", "接下来可以问");
      byId("education-chat-prompts-title").parentElement.after(host);
    }
    host.replaceChildren();
    const questions = learning.suggestions(state.selected, state.lastQuestion, state.lastAnswer || "");
    if (!questions.length) return;
    const title = document.createElement("summary"); title.textContent = "接下来可以问";
    host.append(title);
    questions.forEach(question => {
      const button = document.createElement("button"); button.type = "button";
      button.className = "education-suggested-question"; button.textContent = question;
      button.disabled = !!state.pendingJob || !!state.pendingSubmit || state.sending || state.uploading;
      button.addEventListener("click", () => fillQuestion(question)); host.append(button);
    });
  }

  function openTerm(entry) {
    byId("education-term-title").textContent = entry.name;
    byId("education-term-description").textContent = entry.summary;
    const link = byId("education-term-source"), href = safeSourceUrl(entry.source_url);
    link.classList.toggle("hidden", !href); if (href) link.href = href;
    byId("education-term-read").onclick = () => { byId("education-term-dialog").close(); void selectEntry(entry.id); };
    byId("education-term-dialog").showModal();
  }

  function renderLearningAids(entry) {
    const panel = byId("education-detail");
    const aid = document.createElement("section"); aid.className = "education-learning-aids";
    aid.setAttribute("aria-label", "学习下一步");
    const next = document.createElement("button"); next.type = "button"; next.className = "back-link";
    const available = ["education-explanation", "education-faq-details", "education-media-details"].filter(id => byId(id));
    let step = 0;
    const labels = { "education-explanation": "读一段解释", "education-faq-details": "看常见疑问", "education-media-details": "看专业图文" };
    next.textContent = `下一步：${labels[available[step]] || "结合卡片提问"}`;
    next.addEventListener("click", () => {
      const id = available[step++];
      if (id) { const section = byId(id); section.open = true; section.querySelector("summary").focus(); rememberReading(entry.id, id); }
      else { void importKnowledgeCard(entry.id); }
      next.textContent = `下一步：${labels[available[step]] || "结合卡片提问"}`;
    });
    aid.append(next);
    const terms = learning.glossary(`${entry.summary} ${entry.body}`, state.allEntries, entry.id);
    if (terms.length) {
      const glossary = document.createElement("div"); glossary.className = "education-term-links";
      const caption = document.createElement("span"); caption.textContent = "遇到这些词"; glossary.append(caption);
      terms.forEach(term => { const button = document.createElement("button"); button.type = "button";
        button.textContent = term.name; button.addEventListener("click", () => openTerm(term)); glossary.append(button); });
      aid.append(glossary);
    }
    const quiz = learning.quiz(entry, state.allEntries);
    if (quiz) {
      const details = document.createElement("details"); details.className = "education-mini-review";
      const summary = document.createElement("summary"); summary.textContent = "试一道小复习（可跳过）";
      const prompt = document.createElement("p"); prompt.textContent = `下面这段介绍对应哪个主题？${quiz.prompt}`;
      const answer = document.createElement("p"); answer.setAttribute("role", "status");
      const options = document.createElement("div"); options.className = "education-quiz-options";
      quiz.options.forEach(option => { const button = document.createElement("button"); button.type = "button"; button.textContent = option.name;
        button.addEventListener("click", () => { answer.textContent = option.id === quiz.correctId
          ? `对应「${entry.name}」。${entry.summary}` : `这段介绍对应「${entry.name}」，可以回看简介再试。`;
          button.setAttribute("aria-pressed", "true");
        }); options.append(button); });
      const source = document.createElement("a"); const href = safeSourceUrl(entry.source_url);
      if (href) { source.href = href; source.target = "_blank"; source.rel = "noopener noreferrer"; source.textContent = "查看依据"; }
      details.append(summary, prompt, options, answer, source); aid.append(details);
    }
    // Place learning aids before advanced provenance/tools rather than above the main summary.
    panel.querySelector("#education-source-details").before(aid);
  }

  function saveAnswer(question, answer, citations, mock) {
    if (answer.length > 12000) { byId("education-chat-status").textContent = "这条回答超过本机笔记长度上限，请复制完整回答保存；本次未保存节选。"; return false; }
    if (!window.confirm("将这条问答保存到当前浏览器？同一浏览器的使用者可查看，请勿保存个人病历、身份信息或私密文件内容。")) return false;
    if (learningData.notes.length >= 20) { byId("education-chat-status").textContent = "最多保存 20 条，请先在我的问答中删除部分记录。"; return false; }
    const note = { id: crypto.randomUUID(), entryId: state.selected.id, title: state.selected.name,
      question, answer, citations, mock, date: new Date().toISOString() };
    const updated = { ...learningData, notes: [note, ...learningData.notes] };
    if (!learning.save(preferenceStorage, updated)) {
      byId("education-chat-status").textContent = "浏览器未允许保存或空间不足；本次未保存，可复制当前回答。"; return false;
    }
    learningData.notes = updated.notes;
    byId("education-chat-status").textContent = "已保存到本机，可在我的问答中查看和下载。";
    return true;
  }

  function renderSavedNotes() {
    const list = byId("education-notes-list"); list.replaceChildren();
    if (!learningData.notes.length) { const empty = document.createElement("p"); empty.textContent = "还没有保存问答。回答下方可选择保存到本机。"; list.append(empty); }
    learningData.notes.forEach(note => {
      const details = document.createElement("details"); details.className = "education-saved-note";
      const summary = document.createElement("summary"); summary.textContent = `${note.mock ? "模拟问答 · " : ""}${note.title}：${note.question}`;
      const body = document.createElement("p"); body.textContent = note.answer;
      const links = document.createElement("div");
      note.citations.forEach(citation => { const href = safeSourceUrl(citation.url); if (!href) return;
        const link = document.createElement("a"); link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = citation.title; links.append(link); });
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "quiet-button"; remove.textContent = "删除";
      remove.addEventListener("click", () => {
        const notes = learningData.notes.filter(item => item.id !== note.id);
        if (!learning.save(preferenceStorage, { ...learningData, notes })) { byId("education-notes-status").textContent = "删除未能保存，请检查浏览器存储权限。"; return; }
        learningData.notes = notes; renderSavedNotes();
      });
      const download = document.createElement("button"); download.type = "button"; download.className = "quiet-button"; download.textContent = "下载文本";
      download.addEventListener("click", () => {
        const text = `${note.mock ? "模拟问答，非真实科普结论\n" : ""}${note.title}\n${note.date}\n\n问：${note.question}\n\n答：${note.answer}\n\n来源：\n${note.citations.filter(item => safeSourceUrl(item.url)).map(item => `${item.title}\n${item.url}`).join("\n")}`;
        const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
        const link = document.createElement("a"); link.href = url; link.download = "科普问答.txt"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
      details.append(summary, body, links, download, remove); list.append(details);
    });
  }

  async function prepareHomeQuestion(event) {
    event.preventDefault();
    if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
    const question = byId("education-home-question").value.trim(); if (!question) return;
    const host = byId("education-home-topics"); host.replaceChildren();
    try {
      const catalog = await ensureLearningCatalog(); const matches = learning.matches(question, catalog);
      const exactTopic = matches.some(entry => [entry.name, ...(entry.aliases || [])].some(term => learning.matchedTerm(question, term)));
      const title = document.createElement("p"); title.textContent = exactTopic
        ? "先选一个相关主题，确认后再发送问题。"
        : "尚未准确匹配，可从这些主题入门，也可以改写问题。"; host.append(title);
      matches.forEach(entry => { const button = document.createElement("button"); button.type = "button"; button.className = "quiet-button";
        button.textContent = entry.name;
        button.addEventListener("click", async () => {
          if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
          await selectEntry(entry.id);
          if (state.reading?.id === entry.id) {
            await importKnowledgeCard(entry.id);
            fillQuestion(question);
          }
        }); host.append(button); });
    } catch { host.textContent = "主题暂未加载，请确认本机服务后再试；问题仍保留。"; }
  }

  async function selectEntry(entryId, { switchTopic = false } = {}) {
    if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
    const epoch = ++state.entryEpoch;
    try {
      const entry = await api(`/api/education/entry?id=${encodeURIComponent(entryId)}`);
      await ensureLearningCatalog();
      if (epoch !== state.entryEpoch) return;
      if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
      state.reading = entry;
      state.recentIds = [entry.id, ...state.recentIds.filter((id) => id !== entry.id)].slice(0, 10);
      persistPreferences(); rememberReading(entry.id);
      if (state.selected && !switchTopic) {
        renderEntry(entry); renderPreferenceControls(); renderList();
        setEducationPane("detail", true);
        document.dispatchEvent(new CustomEvent("gea:education-entry-selected", { detail: { entryId: entry.id } }));
        if (state.selected.id !== entry.id) byId("education-chat-status").textContent = `正在阅读「${entry.name}」；对话仍围绕「${state.selected.name}」，草稿和文件已保留。可选择结合这张卡提问。`;
        return;
      }
      if (switchTopic && state.sessionId) {
        try { await api("/api/education/chat/clear", { session_id: state.sessionId }); } catch { /* expired server sessions are already gone */ }
      }
      state.selected = entry;
      state.sessionId = null;
      attachments?.reset();
      state.contextEntries = []; renderCardContext();
      state.pendingJob = null;
      renderEntry(entry);
      renderPreferenceControls();
      renderList();
      setEducationPane("detail", true);
      document.dispatchEvent(new CustomEvent("gea:education-entry-selected", { detail: { entryId: entry.id } }));
      byId("education-chat-topic").textContent = `正在了解：${entry.name}`;
      byId("education-chat-messages").replaceChildren();
      const welcome = document.createElement("p");
      welcome.className = "education-chat-welcome";
      welcome.textContent = `关于「${entry.name}」，你想先弄懂什么？`;
      byId("education-chat-messages").append(welcome);
      if (switchTopic) byId("education-chat-input").value = "";
      state.lastQuestion = ""; state.lastAnswer = ""; renderSuggestedQuestions();
      const canChat = state.modelMode === "mock" || state.modelMode === "ready";
      setChatEnabled(canChat);
      if (state.modelMode === "mock") byId("education-chat-status").textContent = "当前为模拟模式：仅演示对话交互，不调用模型。";
      else if (state.modelMode !== "ready") byId("education-chat-status").textContent = "模型未启用；词条浏览和来源链接仍可使用。";
    } catch (error) {
      byId("education-chat-status").textContent = error.message;
    }
  }

  function clearCurrentSession() {
    const sessionId = state.sessionId;
    state.sessionId = null;
    attachments?.reset();
    state.contextEntries = []; renderCardContext();
    if (sessionId) void api("/api/education/chat/clear", { session_id: sessionId }).catch(() => {});
  }

  async function poll() {
    if (!state.pendingJob || state.polling) return;
    state.polling = true;
    byId("education-chat-resume").classList.add("hidden");
    const startedAt = Date.now();
    byId("education-chat-status").textContent = "正在等待科普助手回复；本机服务不会自动重试失败请求。";
    try {
      while (state.pendingJob && Date.now() - startedAt < 1_200_000) {
        const job = await api(`/api/education/chat/job?session_id=${encodeURIComponent(state.pendingJob.sessionId)}&job_id=${encodeURIComponent(state.pendingJob.jobId)}`);
        if (["completed", "failed"].includes(job.status)) {
          const reply = job.reply;
          const hasAnswer = typeof reply?.answer === "string" && reply.answer.trim().length > 0;
          if (reply?.status === "mock" && hasAnswer) {
            message("assistant", reply.answer, [], true);
            byId("education-chat-status").textContent = "模拟回复 · 没有调用模型，也不是科普结论。";
          } else if (reply?.status === "completed" && hasAnswer) {
            message("assistant", reply.answer, reply.citations || []);
            const usage = reply.usage?.status === "reported" ? ` · 本次报告 ${reply.usage.total_tokens} tokens` : "";
            byId("education-chat-status").textContent = `模型科普整理 · 请结合原始来源核对${usage}`;
          } else {
            if (byId("education-chat-input").value === "" && state.lastQuestion) byId("education-chat-input").value = state.lastQuestion;
            byId("education-chat-status").textContent = reply?.status === "mock" && !hasAnswer
              ? "模拟回复为空；没有调用模型，本次不会自动重试。"
              : errorText[reply?.error_code || (!hasAnswer ? "empty_content" : "")] || "模型回复未能通过检查；请查看来源词条，不会自动重试。";
          }
          state.pendingJob = null;
          byId("education-chat-resume").classList.add("hidden");
          renderList();
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      if (state.pendingJob) {
        byId("education-chat-status").textContent = "仍在等待本机任务；点击继续查看可沿用原任务，不会新增调用。";
        byId("education-chat-resume").classList.remove("hidden");
      }
    } catch (error) {
      byId("education-chat-status").textContent = error.message;
      if (["session_not_found", "job_not_found"].includes(error.code)) {
        state.sessionId = null; state.pendingJob = null; attachments?.reset();
        byId("education-chat-resume").classList.add("hidden");
      } else byId("education-chat-resume").classList.remove("hidden");
    } finally {
      state.polling = false;
      setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
    }
  }

  async function submitPending() {
    if (!state.pendingSubmit) return;
    const submission = state.pendingSubmit;
    try {
      const job = await api("/api/education/chat/message", {
        session_id: submission.sessionId, message_id: submission.messageId,
        entry_id: submission.entryId, text: submission.text, attachment_ids: submission.attachmentIds,
        context_entry_ids: submission.contextEntryIds,
      });
      state.pendingSubmit = null;
      state.pendingJob = { sessionId: submission.sessionId, jobId: job.job_id };
      byId("education-chat-resume").classList.add("hidden");
      setChatEnabled(false);
      await poll();
    } catch (error) {
      if (error.code) {
        if (byId("education-chat-input").value === "") byId("education-chat-input").value = submission.text;
        if (["session_not_found", "attachment_not_found"].includes(error.code)) {
          state.sessionId = null; attachments?.reset();
          byId("education-chat-input").value = submission.text;
        }
        state.pendingSubmit = null;
        byId("education-chat-resume").classList.add("hidden");
        byId("education-chat-status").textContent = error.message;
      } else {
        byId("education-chat-status").textContent = "提交状态暂不可确认。点击“继续确认同一请求”时会沿用原消息 ID，不创建新问题。";
        byId("education-chat-resume").classList.remove("hidden");
      }
    } finally {
      setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
    }
  }

  async function send() {
    const input = byId("education-chat-input");
    const text = input.value.trim();
    if (!text || !state.selected || state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
    state.sending = true;
    setChatEnabled(false);
    try {
      if (!state.sessionId) {
        const session = await api("/api/education/chat/session", { entry_id: state.selected.id });
        state.sessionId = session.session_id;
      }
      const messageId = crypto.randomUUID();
      message("user", text);
      input.value = "";
      state.pendingSubmit = {
        sessionId: state.sessionId, messageId,
        entryId: state.selected.id, text, attachmentIds: attachments?.ids() || [],
        contextEntryIds: (state.contextEntries || []).map((entry) => entry.id),
      };
      setChatEnabled(false);
      await submitPending();
    } catch (error) {
      byId("education-chat-status").textContent = error.message;
    } finally {
      state.sending = false;
      setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
    }
  }

  byId("research-toggle").addEventListener("click", () => updateNavigation(false));
  byId("education-toggle").addEventListener("click", () => updateNavigation(true));
  byId("education-return").addEventListener("click", () => updateNavigation(false));
  document.querySelectorAll("[data-learning-kind]").forEach((button) => button.addEventListener("click", () => {
    if (state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
    document.querySelector(`[data-education-kind="${button.dataset.learningKind}"]`)?.click();
    setEducationPane("catalog", true);
  }));
  document.querySelectorAll(".education-mobile-nav [data-education-pane]").forEach((button) => button.addEventListener("click", () => {
    if (!button.disabled) setEducationPane(button.dataset.educationPane, true);
  }));
  document.querySelectorAll("[data-education-kind]").forEach((button) => button.addEventListener("click", () => {
    if (state.pendingJob || state.pendingSubmit || state.sending) return;
    state.kind = button.dataset.educationKind;
    state.query = "";
    byId("education-search").value = "";
    state.category = "";
    // Changing the catalog filter must never reset the active conversation.
    setEducationPane("catalog");
    document.querySelectorAll("[data-education-kind]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    setChatEnabled(!!state.selected && (state.modelMode === "mock" || state.modelMode === "ready"));
    void loadCatalog();
  }));
  byId("education-search").addEventListener("input", (event) => {
    state.query = event.target.value.trim();
    state.category = "";
    void loadCatalog();
  });
  byId("education-category").addEventListener("change", (event) => { state.category = event.target.value; renderCategoryFilters(); renderList(); });
  byId("education-filter-reset").addEventListener("click", () => {
    state.query = ""; state.category = ""; state.favoritesOnly = false; state.recentOnly = false;
    byId("education-search").value = "";
    renderPreferenceControls();
    void loadCatalog();
  });
  byId("education-favorites-toggle").addEventListener("click", () => {
    state.favoritesOnly = !state.favoritesOnly;
    if (state.favoritesOnly) state.recentOnly = false;
    renderPreferenceControls();
    renderList();
  });
  byId("education-recent-toggle").addEventListener("click", () => {
    state.recentOnly = !state.recentOnly;
    if (state.recentOnly) state.favoritesOnly = false;
    renderPreferenceControls();
    renderList();
  });
  byId("education-preferences-reset").addEventListener("click", () => {
    if (!window.confirm("清除本机保存的科普收藏、最近阅读和字号设置？这不会清除研究查询历史或服务器数据。")) return;
    const cleared = preferenceStore.clear(preferenceStorage);
    state.favoriteIds.clear();
    state.recentIds = [];
    learningData.progress = null; learning.save(preferenceStorage, learningData); refreshContinueButton();
    state.readerSize = "standard";
    state.favoritesOnly = false;
    state.recentOnly = false;
    byId("education-preferences-status").textContent = cleared
      ? "已清除本机科普收藏、阅读记录和字号设置。"
      : "浏览器未允许清除保存设置；本页面已恢复默认状态。";
    renderPreferenceControls();
    renderList();
  });
  byId("education-chat-send").addEventListener("click", () => void send());
  document.querySelectorAll(".education-chat-prompt").forEach((button) => {
    button.addEventListener("click", () => { applyEducationPrompt(button.dataset.educationPrompt); });
  });
  byId("education-chat-resume").addEventListener("click", () => {
    if (state.pendingSubmit) void submitPending();
    else void poll();
  });
  byId("education-chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) void send();
  });
  byId("education-chat-clear").addEventListener("click", async () => {
    if (!state.sessionId || state.pendingJob || state.pendingSubmit || state.sending || state.uploading) return;
    try { await api("/api/education/chat/clear", { session_id: state.sessionId }); } catch { /* expiration also clears the local binding */ }
    state.sessionId = null;
    attachments?.reset();
    state.contextEntries = []; renderCardContext(); renderList();
    byId("education-chat-messages").replaceChildren();
    state.lastQuestion = ""; state.lastAnswer = ""; renderSuggestedQuestions();
    byId("education-chat-status").textContent = "新科普对话已准备。";
    setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
  });

  byId("education-home-question-form").addEventListener("submit", event => void prepareHomeQuestion(event));
  byId("education-home-open").addEventListener("click", () => {
    setEducationPane("catalog"); byId("education-view").dataset.catalogOpen = "false";
    byId("education-home-question").focus();
  });
  byId("education-continue-learning").addEventListener("click", async () => {
    const id = byId("education-continue-learning").dataset.entryId;
    if (!id) return;
    const sectionId = learningData.progress?.section;
    await selectEntry(id);
    if (state.reading?.id !== id) return;
    const section = sectionId && byId(sectionId);
    if (section?.tagName === "DETAILS") { section.open = true; section.querySelector("summary").focus(); }
  });
  byId("education-notes-open").addEventListener("click", () => { renderSavedNotes(); byId("education-notes-status").textContent = ""; byId("education-notes-dialog").showModal(); });
  byId("education-notes-close").addEventListener("click", () => byId("education-notes-dialog").close());
  byId("education-term-close").addEventListener("click", () => byId("education-term-dialog").close());
  byId("education-notes-clear").addEventListener("click", () => {
    if (!window.confirm("清空所有主动保存的问答？此操作不能撤销。")) return;
    if (!learning.save(preferenceStorage, { ...learningData, notes: [] })) { byId("education-notes-status").textContent = "清空未能保存，请检查浏览器存储权限。"; return; }
    learningData.notes = []; renderSavedNotes();
  });

  updateNavigation(true, { focusSearch: false });
  setEducationPane("catalog");
  renderPreferenceControls();
  if (!preferenceStorage) byId("education-preferences-status").textContent = "浏览器未允许本机保存；收藏、阅读记录和字号仍可在当前页面使用。";
  void api("/api/health").then((health) => {
    state.modelMode = health.mode === "mock" ? "mock" : health.education?.chat?.enabled === true ? "ready" : "disabled";
    if (state.selected) setChatEnabled(state.modelMode === "mock" || state.modelMode === "ready");
  }).catch(() => { state.modelMode = "disabled"; });
  void loadCatalog();
  void ensureLearningCatalog().catch(() => {});
})();
