(() => {
  const byId = (id) => document.getElementById(id);
  const STORAGE_KEY = "gea.education.guide.v1";
  const quickStepIds = new Set(["choose-kind", "search", "select-entry"]);
  const steps = [
    {
      id: "choose-kind", selector: "#education-kind-tabs", pane: "catalog",
      title: "先选一个方向", text: "选择癌症、基因或基础知识，按你的兴趣开始。",
      art: { src: "/education-guides/GUIDE-02-web.jpg", alt: "癌症细胞、DNA 双螺旋和医学词语书籍的抽象图形，代表三个学习入口。" },
    },
    {
      id: "search", selector: "#education-search", pane: "catalog",
      title: "也可以直接搜索", text: "输入中文、英文或你听到的名称，不必记住标准术语。",
      art: { src: "/education-guides/GUIDE-03-web.jpg", alt: "知识向导旁有空白搜索框和对话气泡，表示可以用日常语言提问。" },
    },
    {
      id: "select-entry", selector: "#education-entry-list", pane: "catalog",
      title: "打开一个主题", text: "点选列表中的任意主题。选好后，引导会接着介绍阅读页面。",
      needsEntryAction: true,
    },
    {
      id: "read-summary", selector: "#education-detail .education-entry-summary", pane: "detail",
      title: "先看简短说明", text: "从这一段先了解大意；想看更多时，再继续读下面的介绍。",
      needsSelection: true,
    },
    {
      id: "see-media", selector: "#education-detail .education-media-section", pane: "detail",
      title: "按需看图或视频", text: "只有经过核对并可用的资料才会显示在这里。当前主题没有媒体时，可以直接继续。",
      needsSelection: true,
      art: { src: "/education-guides/GUIDE-04-web.jpg", alt: "打开的书、示意图和带播放符号的屏幕，表示可以阅读、看图或视频。" },
    },
    {
      id: "ask", selector: "#education-chat", pane: "chat",
      title: "用日常话继续问", text: "点建议问题只会填入草稿，确认发送后才会提交。",
      needsSelection: true,
      art: { src: "/education-guides/GUIDE-05-web.jpg", alt: "知识向导与两个对话气泡，表示可以继续追问并阅读解释。" },
    },
    {
      id: "sources", selector: "#education-detail .education-provenance", pane: "detail",
      title: "需要时查看出处", text: "这里显示来源与核对状态。来源核对不等于独立医学专家审阅。",
      needsSelection: true,
      art: { src: "/education-guides/GUIDE-06-web.jpg", alt: "知识向导展示书签和相连的资料卡，表示可继续查看信息出处。" },
    },
  ];

  let storage = null;
  try { storage = window.localStorage; } catch { /* Browser storage may be unavailable. */ }
  let guideIndex = -1;
  let guideTrigger = null;
  let guideMode = "full";
  let guideEpoch = 0;
  let hasSelection = !!document.querySelector("#education-detail .education-entry-title");

  byId("education-guide-art").addEventListener("error", () => {
    byId("education-guide-art-figure").classList.add("hidden");
  });

  byId("education-onboarding-illustration").addEventListener("error", () => {
    byId("education-onboarding-illustration").classList.add("hidden");
    byId("education-onboarding-art-fallback").classList.remove("hidden");
  });

  function savedStatus() {
    try { return storage?.getItem(STORAGE_KEY) || ""; } catch { return ""; }
  }

  function persistStatus(status) {
    try { storage?.setItem(STORAGE_KEY, status); } catch { /* The guide still works without persistence. */ }
  }

  function hideIntroIfCompleted() {
    byId("education-onboarding").classList.toggle("hidden", ["completed", "skipped"].includes(savedStatus()));
  }

  function availableSteps() {
    return steps.filter((step) => {
      if (guideMode === "quick") {
        const quickIds = hasSelection ? new Set(["read-summary"]) : quickStepIds;
        if (!quickIds.has(step.id)) return false;
      }
      if (step.needsSelection && !hasSelection) return false;
      if (step.needsEntryAction && hasSelection) return false;
      const target = document.querySelector(step.selector);
      return !!target && !target.hidden && !target.closest("[hidden], .hidden");
    });
  }

  function removeHighlights() {
    document.querySelectorAll(".education-guide-target").forEach((element) => element.classList.remove("education-guide-target"));
  }

  function paneIsVisible(step) {
    if (!step.pane) return true;
    return document.querySelector(`.education-mobile-nav [data-education-pane="${step.pane}"]`)?.getAttribute("aria-pressed") === "true";
  }

  function renderGuide() {
    if (guideIndex < 0) return;
    const visibleSteps = availableSteps();
    if (guideIndex >= visibleSteps.length) {
      if (guideMode === "full" && hasSelection && guideIndex > 0 && guideIndex >= visibleSteps.length) showGuideFinish();
      else closeGuide("completed");
      return;
    }

    byId("education-guide-finish").classList.add("hidden");
    byId("education-guide-next").classList.remove("hidden");
    byId("education-guide-skip").textContent = "跳过";

    const step = visibleSteps[guideIndex];
    if (!paneIsVisible(step)) {
      const paneButton = document.querySelector(`.education-mobile-nav [data-education-pane="${step.pane}"]`);
      if (paneButton && !paneButton.disabled) {
        paneButton.click();
        const epoch = guideEpoch;
        window.requestAnimationFrame(() => {
          if (epoch === guideEpoch && guideIndex >= 0) renderGuide();
        });
        return;
      }
    }

    const target = document.querySelector(step.selector);
    if (!target) {
      guideIndex += 1;
      renderGuide();
      return;
    }

    removeHighlights();
    // Keep progressive reading closed until a guide explicitly visits that section.
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === "DETAILS") parent.open = true;
    }
    target.classList.add("education-guide-target");
    const phaseSteps = visibleSteps.filter((candidate) => Boolean(candidate.needsSelection) === Boolean(step.needsSelection));
    const phaseIndex = phaseSteps.findIndex((candidate) => candidate.id === step.id);
    byId("education-guide-count").textContent = `${phaseIndex + 1} / ${phaseSteps.length}`;
    byId("education-guide-title").textContent = step.title;
    byId("education-guide-text").textContent = step.text;
    const guideArt = byId("education-guide-art");
    const guideArtFigure = byId("education-guide-art-figure");
    if (step.art) {
      guideArt.setAttribute("src", step.art.src);
      guideArt.setAttribute("alt", step.art.alt);
      guideArtFigure.classList.remove("hidden");
    } else {
      guideArtFigure.classList.add("hidden");
    }
    byId("education-guide-prev").disabled = guideIndex === 0;
    const isQuickReadFinish = guideMode === "quick" && step.id === "read-summary";
    const isLastStep = guideIndex === visibleSteps.length - 1;
    byId("education-guide-next").textContent = step.needsEntryAction && !hasSelection
      ? "先选主题"
      : isQuickReadFinish
        ? "完成快速入门"
        : isLastStep ? "选择下一步" : "下一步";
    byId("education-guide-next").disabled = step.needsEntryAction && !hasSelection;
    byId("education-guide-popover").classList.remove("hidden");
    target.scrollIntoView?.({ block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    byId("education-guide-next").focus({ preventScroll: true });
  }

  function showGuideFinish() {
    removeHighlights();
    byId("education-guide-count").textContent = "完成";
    byId("education-guide-title").textContent = "接下来，由你选择";
    byId("education-guide-text").textContent = "继续读、问一问，或换个主题。";
    byId("education-guide-art-figure").classList.add("hidden");
    byId("education-guide-next").classList.add("hidden");
    byId("education-guide-skip").textContent = "先到这里";
    byId("education-guide-finish").classList.remove("hidden");
    byId("education-guide-popover").classList.remove("hidden");
    byId("education-guide-continue").focus({ preventScroll: true });
  }

  function closeGuide(status = "skipped", restoreFocus = true, persist = true) {
    const wasOpen = guideIndex >= 0;
    const closedEpoch = ++guideEpoch;
    guideIndex = -1;
    byId("education-guide-popover").classList.add("hidden");
    byId("education-guide-finish").classList.add("hidden");
    byId("education-guide-next").classList.remove("hidden");
    byId("education-guide-skip").textContent = "跳过";
    removeHighlights();
    if (persist && ["completed", "skipped"].includes(status)) persistStatus(status);
    hideIntroIfCompleted();
    if (status === "completed") byId("education-guide-status").textContent = "体验完成。你可以随时重新打开新手引导。";
    else if (status === "skipped") byId("education-guide-status").textContent = "已关闭引导；主题介绍和来源仍可直接浏览。";
    else if (status === "cancelled") byId("education-guide-status").textContent = "引导已暂停；需要时可以重新开始。";
    if (restoreFocus && wasOpen) {
      const triggerVisible = guideTrigger && !guideTrigger.disabled && !guideTrigger.hidden && !guideTrigger.closest("[hidden], .hidden");
      const returnTarget = triggerVisible ? guideTrigger : byId("education-guide-restart");
      returnTarget?.focus({ preventScroll: true });
      window.requestAnimationFrame(() => {
        if (guideEpoch === closedEpoch && guideIndex < 0) returnTarget?.focus({ preventScroll: true });
      });
    }
    guideTrigger = null;
  }

  function startGuide(trigger, mode = "full") {
    guideEpoch += 1;
    guideTrigger = trigger || byId("education-guide-restart");
    byId("education-onboarding").classList.add("hidden");
    hasSelection = !!document.querySelector("#education-detail .education-entry-title");
    guideMode = mode === "quick" ? "quick" : "full";
    guideIndex = 0;
    document.querySelector(`.education-mobile-nav [data-education-pane="${hasSelection ? "detail" : "catalog"}"]`)?.click();
    byId("education-guide-status").textContent = guideMode === "quick"
      ? "快速入门已开始。只介绍找到并打开主题的步骤，不会调用模型。"
      : "完整导览已开始。每一步都可以跳过，且不会自动发送问题。";
    renderGuide();
  }

  function nextGuideStep() {
    const visibleSteps = availableSteps();
    const current = visibleSteps[guideIndex];
    if (!current) return;
    if (current.needsEntryAction && !hasSelection) {
      byId("education-guide-status").textContent = "请先从主题列表中选择一项；选好后会自动继续。";
      return;
    }
    guideIndex += 1;
    renderGuide();
  }

  function previousGuideStep() {
    guideIndex = Math.max(0, guideIndex - 1);
    renderGuide();
  }

  function finishAndFocus(pane, targetId, message) {
    closeGuide("completed", false);
    const epoch = guideEpoch;
    byId("education-guide-status").textContent = message;
    if (pane) {
      document.querySelector(`.education-mobile-nav [data-education-pane="${pane}"]`)?.click();
    }
    window.requestAnimationFrame(() => {
      if (guideEpoch === epoch && guideIndex < 0) byId(targetId)?.focus({ preventScroll: true });
    });
  }

  byId("education-onboarding-quick").addEventListener("click", (event) => startGuide(event.currentTarget, "quick"));
  byId("education-onboarding-start").addEventListener("click", (event) => startGuide(event.currentTarget, "full"));
  byId("education-guide-restart").addEventListener("click", (event) => startGuide(event.currentTarget));
  byId("education-onboarding-browse").addEventListener("click", () => {
    persistStatus("skipped");
    hideIntroIfCompleted();
    document.querySelector('.education-mobile-nav [data-education-pane="catalog"]')?.click();
    byId("education-search").focus({ preventScroll: true });
    byId("education-guide-status").textContent = "可以直接浏览主题；需要帮助时，点击“新手引导”。";
  });
  byId("education-guide-next").addEventListener("click", nextGuideStep);
  byId("education-guide-prev").addEventListener("click", previousGuideStep);
  byId("education-guide-skip").addEventListener("click", () => {
    if (guideMode === "full" && hasSelection && guideIndex > 0 && guideIndex >= availableSteps().length) closeGuide("completed");
    else closeGuide("skipped");
  });
  byId("education-guide-continue").addEventListener("click", () => finishAndFocus("detail", "education-detail", "引导已完成。你可以继续阅读当前主题，也可以随时重看新手引导。"));
  byId("education-guide-ask").addEventListener("click", () => finishAndFocus("chat", "education-chat-input", "已打开当前主题的追问区；检查问题后点击发送，才会请求模型。"));
  byId("education-guide-switch").addEventListener("click", () => finishAndFocus("catalog", "education-search", "已回到主题目录，可以搜索或浏览其他内容。"));

  document.addEventListener("gea:education-entry-selected", () => {
    hasSelection = true;
    if (guideIndex < 0) return;
    if (guideMode === "quick") guideIndex = 0;
    renderGuide();
  });
  document.addEventListener("gea:education-entry-cleared", () => {
    hasSelection = false;
    if (guideIndex < 0) return;
    guideIndex = Math.max(0, availableSteps().findIndex((step) => step.id === "select-entry"));
    renderGuide();
  });
  document.addEventListener("gea:education-view-hidden", () => closeGuide("cancelled", false, false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && guideIndex >= 0) closeGuide("skipped");
  });

  hideIntroIfCompleted();
  window.GeaEducationOnboarding = Object.freeze({ start: startGuide, stop: closeGuide });
})();
