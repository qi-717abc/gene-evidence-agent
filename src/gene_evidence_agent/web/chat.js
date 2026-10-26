/* Server jobs settle usage even when the tab stops waiting. */
(() => {
  // The confirmation-gated workflow owns the right panel when present.
  if (document.querySelector("[data-guided-workflow]")) return;
  const n = (id) => document.getElementById(id);
  const s = { session: null, latest: null, bound: undefined, pending: null, enabled: false, polling: false, paused: false, creating: false };
  const errors = { session_not_found: "会话已过期，请建立新会话。", report_not_found: "证据已过期，请重新检索。", session_busy: "上一请求仍在处理。", context_change_requires_new_session: "请建立新会话切换上下文。", sensitive_query_rejected: "请移除个人信息，使用公开研究问题。", model_disabled: "服务端未启用对话模型。", model_budget: "预算不足，未发送请求。", timeout: "模型超时，未自动重试。", output_truncated: "模型回复截断，未接受该回答。", invalid_citation: "回复引用未通过校验。", invalid_model_output: "回复结构未通过校验。", unknown_usage: "没有可信用量，预留仍保留。", rate_limited: "服务限流，未自动重试。", empty_content: "模型正文为空。", transport_error: "模型连接失败，预留仍保留。", model_error: "模型请求失败。", context_too_large: "上下文过大，请新建会话。", capacity_reached: "服务繁忙，请稍后再试。", session_turn_limit: "会话已达轮次上限。" };
  function toggle(open) {
    document.body.classList.toggle("chat-open", open);
    document.body.classList.toggle("chat-collapsed", !open);
    n("chat-toggle").setAttribute("aria-expanded", String(open));
  }
  errors.input_token_preflight_mismatch = "服务商输入用量超过本地预留边界，回答未接受；已返回用量仍计入账本。";
  errors.output_token_limit_exceeded = "服务商报告的输出用量超过本次请求上限；结果未接受，已返回用量仍计入账本。";
  errors.synthesis_not_available = "综合报告尚未完成、已过期或不属于此检索版本。";
  errors.unsupported_claim = "回复的研究设计、信息缺失、术语或因果措辞未通过依据校验，回答未接受；原始证据保留。";
  const citationDiagnostics = {unknown_id:"引用含本次未提供的ID",malformed_marker:"引用标记格式不符",markers_mismatch:"正文引用与引用列表不一致",missing_citation:"缺少必需引用",unexpected_url:"正文出现未经许可的网址"};
  const semanticDiagnostics = {unsupported_study_design:"研究设计缺少明确原句",unsupported_absence_claim:"把摘要未报告推成研究未实施",source_terminology_mismatch:"术语上下文不符",source_causality_mismatch:"因果措辞超出原句"};
  function controls() {
    const busy = !!s.pending || s.creating;
    n("chat-input").disabled = !s.enabled || busy;
    n("chat-send").disabled = !s.enabled || busy;
    n("chat-clear").disabled = busy;
    n("chat-use-evidence").disabled = busy || s.bound !== undefined || !s.latest?.result_id;
    n("chat-stop").classList.toggle("hidden", !s.pending || s.paused);
    n("chat-resume").classList.toggle("hidden", !s.pending || !s.paused);
    n("chat-context-label").textContent = s.bound === undefined ? (s.latest?.result_id ? "可用结果：" + s.latest.query : "基础知识问答；检索后可加入证据。") : s.bound ? (s.bound.synthesis_id ? "绑定综合报告：" : "绑定结果：") + s.bound.query : "基础知识会话；新会话可加入检索结果。";
  }
  async function api(path, data) {
    const response = await fetch(path, data === undefined ? { cache: "no-store" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const value = await response.json();
    if (!response.ok) { const error = new Error(errors[value.error] || "本机接口拒绝请求。"); error.definitive = true; error.code = value.error; throw error; }
    return value;
  }
  function message(role, text, reply) {
    n("chat-messages").querySelector(".chat-empty")?.remove();
    const card = document.createElement("article");
    card.className = "chat-message chat-message-" + role;
    const label = document.createElement("div"); label.className = "chat-message-role";
    label.textContent = role === "user" ? "你" : reply?.status === "mock" ? "模拟助手 · 未调用模型" : reply?.answer_generation_method === "report_review_checklist" ? "原论文核对清单 · 未调用模型" : reply?.answer_generation_method === "report_extractive_fallback" ? "原报告摘录 · 模型回答未通过校验" : reply?.answer_generation_method === "report_fact_extractive" ? "研究卡事实摘录 · 未调用模型" : "模型回答 · 引用可核验";
    const body = document.createElement("div"); body.className = "chat-message-text"; body.textContent = text;
    card.append(label, body);
    if (reply) {
      const sources = document.createElement("div"); sources.className = "chat-citations";
      for (const ref of reply.citations || []) {
        try {
          const url = new URL(ref.url);
          if (url.protocol !== "https:" || url.username || url.password) continue;
          const link = document.createElement("a"); link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer";
          link.textContent = "[" + ref.source_id + "] " + (ref.kind === "knowledge" ? "知识来源：" : "检索证据：") + ref.title;
          sources.append(link);
          if (ref.kind === "evidence" && Number.isInteger(ref.record_index) && ref.record_index >= 0 && reply.result_id === s.latest?.result_id) {
            const locate = document.createElement("button"); locate.type = "button"; locate.className = "back-link";
            locate.textContent = "定位左侧证据 [" + ref.source_id + "]";
            locate.addEventListener("click", () => {
              n("results-view").classList.remove("hidden"); n("search-view").classList.add("hidden");
              document.getElementById("chat-evidence-" + ref.record_index)?.scrollIntoView({ behavior: "smooth", block: "center" });
            });
            sources.append(locate);
          }
        } catch { /* Discard unsafe links. */ }
      }
      const meta = document.createElement("p"); meta.className = "chat-caption";
      const contexts = { mock: "模拟上下文", general: "模型一般解释（未来源核实）", knowledge: "知识库资料", evidence: "检索证据", knowledge_and_evidence: "知识库与检索证据", synthesis: "逐篇对照报告与原始摘要" };
      meta.textContent = (contexts[reply.context_kind] || "上下文未知") + " · 知识版本 " + reply.knowledge_version + " · " + (reply.usage.status === "reported" ? reply.usage.total_tokens + " tokens" : reply.usage.status === "not_called" ? "未调用模型" : "用量未知，预留保留");
      card.append(sources, meta);
    }
    n("chat-messages").append(card); n("chat-messages").scrollTop = n("chat-messages").scrollHeight;
  }
  async function poll() {
    if (s.polling || !s.pending) return;
    s.polling = true; s.paused = false; controls();
    try {
      if (!s.pending.job) s.pending.job = (await api("/api/chat/message", s.pending.request)).job_id;
      while (s.pending && !s.paused) {
        const job = await api("/api/chat/job?session_id=" + encodeURIComponent(s.session) + "&job_id=" + encodeURIComponent(s.pending.job));
        if (job.reply) {
          if (job.status === "completed") message("assistant", job.reply.answer, job.reply);
          const use = job.reply.usage;
          const usageNote = use.status === "reported" ? " 已返回 " + use.total_tokens + " tokens，用量已记录。" : use.status === "unknown" ? " 用量未知，预留保留。" : " 未调用模型。";
          const diagnostic=citationDiagnostics[job.reply.citation_diagnostic]||semanticDiagnostics[job.reply.semantic_diagnostic];
          const ruleLabels={review:"综述",in_vitro:"体外实验",in_vivo:"体内实验",clinical_trial:"临床试验",unperformed_assessment:"未实施评估",validation_absence:"验证不存在",category_absence:"类别不存在"};
          const ruleNote=ruleLabels[job.reply.semantic_rule] ? "；规则："+ruleLabels[job.reply.semantic_rule] : "";
          const diagnosticNote=diagnostic ? "（"+diagnostic+ruleNote+"）" : "";
          n("chat-status").textContent = job.status === "failed" ? (errors[job.reply.error_code] || "生成失败。") + diagnosticNote + usageNote : job.reply.answer_generation_method === "report_extractive_fallback" ? "已回退到原报告摘录；模型自由回答未通过校验。" + diagnosticNote + usageNote : job.reply.answer_generation_method === "report_fact_extractive" ? "已从当前报告绑定的研究卡摘录原文；未调用模型。" + diagnosticNote + usageNote : "回复完成。";
          s.pending = null; break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    } catch (error) {
      if (error.definitive && (!s.pending?.job || ["session_not_found", "job_not_found"].includes(error.code))) { s.pending = null; n("chat-status").textContent = error.message; }
      else { s.paused = true; n("chat-status").textContent = "读取状态中断。点击继续查看，以同一消息ID确认，不会建立重复请求。"; }
    } finally { s.polling = false; controls(); if (!s.pending) void configure(); }
  }
  async function send() {
    const text = n("chat-input").value.trim();
    if (!text || !s.enabled || s.pending || s.creating) return;
    s.creating = true; controls();
    try {
      if (!s.session) s.session = (await api("/api/chat/session", {})).session_id;
      if (s.bound === undefined) s.bound = n("chat-use-evidence").checked ? s.latest : null;
      s.pending = { job: null, request: { session_id: s.session, message_id: crypto.randomUUID(), text, use_evidence: !!s.bound, result_id: s.bound?.result_id || null } };
      if (s.bound?.synthesis_id) s.pending.request.synthesis_id = s.bound.synthesis_id;
      message("user", text); n("chat-input").value = ""; n("chat-status").textContent = "正在等待回复…"; controls(); await poll();
    } catch (error) { n("chat-status").textContent = error.message || "创建会话失败。"; }
    finally { s.creating = false; controls(); }
  }
  n("chat-toggle").addEventListener("click", () => toggle(n("chat-toggle").getAttribute("aria-expanded") !== "true"));
  n("chat-close").addEventListener("click", () => toggle(false));
  n("chat-send").addEventListener("click", () => void send());
  n("chat-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) void send(); });
  n("chat-stop").addEventListener("click", () => { s.paused = true; n("chat-status").textContent = "已停止等待；后台请求继续结算，结束前不能重发或清空。"; controls(); });
  n("chat-resume").addEventListener("click", () => void poll());
  n("chat-clear").addEventListener("click", async () => {
    if (s.pending) return;
    try { if (s.session) await api("/api/chat/clear", { session_id: s.session }); }
    catch (error) { if (error.code !== "session_not_found") { n("chat-status").textContent = error.message; return; } }
    s.session = null; s.bound = undefined; n("chat-messages").replaceChildren(); n("chat-status").textContent = "新会话已准备。"; controls();
  });
  document.querySelectorAll("[data-chat-example]").forEach((button) => button.addEventListener("click", () => { if (!s.pending) n("chat-input").value = button.dataset.chatExample; }));
  window.addEventListener("gene-evidence:result", (event) => {
    s.latest = event.detail;
    if (s.bound !== undefined) n("chat-status").textContent = "新检索完成；当前会话仍使用旧上下文。新会话可切换结果。";
    controls();
  });
  window.addEventListener("gene-evidence:synthesis-followup", async (event) => {
    if (s.pending || s.creating) { n("chat-status").textContent = "当前请求结束后才能切换综合报告。"; return; }
    try {
      if (s.session) await api("/api/chat/clear", {session_id:s.session});
    } catch (error) { if (error.code !== "session_not_found") { n("chat-status").textContent = error.message; return; } }
    s.session = null; s.bound = event.detail; n("chat-messages").replaceChildren();
    n("chat-use-evidence").checked = true; n("chat-input").value = "这份综合报告最重要的证据缺口是什么？";
    n("chat-status").textContent = "已绑定此综合报告，发送后可连续追问。"; toggle(true); await configure();
  });
  async function configure() {
    try {
      const health = await api("/api/health"); s.enabled = s.bound?.synthesis_id ? health.chat?.synthesis_enabled === true : health.chat?.enabled === true;
      n("chat-mode").textContent = health.chat?.mode === "mock" ? "模拟对话 · 不调用模型" : s.enabled ? "领域模型对话 · 资料来源可核验" : health.chat?.status === "model_budget" ? "对话预算或验收请求名额暂不可用；检索仍可使用。" : "服务端未启用对话模型；检索仍可使用。";
    } catch { n("chat-mode").textContent = "本机对话服务不可用。"; }
    controls();
  }
  if (window.matchMedia("(max-width: 1100px)").matches) toggle(false);
  void configure();
})();
