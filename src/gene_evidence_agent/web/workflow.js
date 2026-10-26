/* The left panel is the confirmation-gated research conversation. */
(() => {
  if (!document.querySelector("[data-guided-workflow]")) return;
  const byId = (id) => document.getElementById(id);
  const welcome = "你好，我是基因—癌症研究助手。可以检索公开证据、查看英文论文。先告诉我基因和疾病，例如“TP53 与肺癌”。";
  const state = { session: null, job: null, polling: false, paused: false, sending: false, snapshot: null, renderedResultId: null };
  const sessionKey = "gene-evidence-workflow-session-v1";
  const attachments = window.GeaAttachments?.create({
    input: "workflow-input", channel: "workflow", getSession: async () => state.session,
    canUpload: () => !!state.session && !state.sending && !state.job,
    onBusy: (busy) => { state.uploading = busy; setControls(); }
  });
  const errors = {
    session_not_found: "对话已过期，请新建研究对话。", sensitive_query_rejected: "请移除患者身份或个人病历信息，改用公开研究问题。",
    session_busy: "上一条消息仍在处理。", action_stale: "确认卡已过期，请重新检查当前条件。", action_not_found: "找不到该操作，请重新准备。",
    operation_in_progress: "另一个已确认的操作仍在执行。", result_required: "请先完成一次检索。", result_not_found: "该结果版本已过期。",
    insufficient_abstracts: "本次结果少于两篇有英文摘要的论文，不足以生成逐篇对照。", localization_disabled: "服务端未启用中文整理。",
    synthesis_disabled: "服务端未启用综合。", drug_option_requires_opentargets: "只有选择 Open Targets 才能包含疾病层级药物候选。",
    unsupported_chinese_disease_term: "这个中文疾病名称暂未收录可靠英文映射，尚未检索。请改填英文疾病名称后再确认。",
    english_gene_symbol_required: "请使用英文基因符号（例如 TP53），当前没有启动检索。",
    interpretation_failed: "需求解析失败；没有执行任何检索。请用“基因与癌症”的形式再描述一次。", invalid_request: "请求内容无效。",
    capacity_reached: "本机服务繁忙。", session_turn_limit: "本研究对话达到本地轮次上限，请新建会话。", workflow_failed: "工作流未完成；之前结果仍保留。",
    context_too_large: "报告与原始摘要上下文过大，未发送模型请求；请缩小报告范围后再试。",
    model_budget: "模型预算不足，未发送请求。", model_error: "模型请求失败；没有自动重试。",
    timeout: "模型请求超时；没有自动重试。", rate_limited: "模型服务限流；没有自动重试。",
    transport_error: "模型连接失败；没有自动重试。", output_truncated: "模型回复被截断，未接受该回答。",
    output_token_limit_exceeded: "服务商报告的输出用量超过本次请求上限；结果未接受，已返回用量仍计入账本。",
    empty_content: "模型返回空正文，未接受该回答。", unknown_usage: "模型用量未知；保留预留且未继续请求。",
    invalid_model_output: "模型回复结构未通过校验，未接受该回答。", invalid_citation: "模型回复的来源引用未通过校验。",
    unsupported_claim: "模型回答的事实或措辞超出来源，未接受；请查看报告中的已核验内容。",
    staged_synthesis_required: "请在逐篇文献对照面板逐步确认规划、研究卡和报告。"
  };
  const names = { retrieve: "检索", localize: "中文整理", synthesize: "英文跨文献报告" };

  async function api(path, body) {
    const response = await fetch(path, body === undefined ? { cache: "no-store" } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
    });
    const value = await response.json();
    if (!response.ok) { const error = new Error(errors[value.error] || "本机接口暂不可用（" + (value.error || response.status) + "）。"); error.code = value.error; throw error; }
    return value;
  }
  function node(tag, text, className) { const el = document.createElement(tag); if (text) el.textContent = text; if (className) el.className = className; return el; }
  function setChatOpen(open, focusInput = false) {
    document.body.classList.toggle("chat-open", open);
    document.body.classList.toggle("chat-collapsed", !open);
    byId("chat-toggle").setAttribute("aria-expanded", String(open));
    byId("chat-toggle").textContent = open ? "收起对话" : "打开对话";
    if (open && focusInput) requestAnimationFrame(() => byId("workflow-input").focus());
  }
  function setComposerText(text) {
    byId("workflow-input").value = text;
    byId("workflow-char-count").textContent = text.length + "/3000";
  }
  function roleLabel(role) { return role === "user" ? "你" : "研究助手"; }
  function appendMessage(role, text, reply) {
    const card = node("article", null, "chat-message chat-message-" + (role === "user" ? "user" : "assistant"));
    card.append(node("div", roleLabel(role), "chat-message-role"), node("div", text || "", "chat-message-text"));
    if (reply) {
      const citations = node("div", null, "chat-citations");
      for (const citation of reply.citations || []) {
        try {
          const url = new URL(citation.url);
          if (url.protocol !== "https:" || url.username || url.password) continue;
          const link = node("a", "[" + citation.source_id + "] " + citation.title);
          link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer"; citations.append(link);
          if (citation.kind === "evidence" && Number.isInteger(citation.record_index)
              && reply.result_id === state.snapshot?.current_result?.result_id) {
            const locate = node("button", "定位左侧证据", "back-link"); locate.type = "button";
            locate.addEventListener("click", () => {
              byId("results-view").classList.remove("hidden"); byId("search-view").classList.add("hidden");
              document.getElementById("chat-evidence-" + citation.record_index)?.scrollIntoView({ behavior: "smooth", block: "center" });
            });
            citations.append(locate);
          }
        } catch { /* Unsafe citations are not rendered as links. */ }
      }
      if (citations.childNodes.length) card.append(citations);
      const usage = reply.usage?.status === "reported" ? "用量 " + reply.usage.total_tokens + " tokens" : reply.usage?.status === "not_called" ? "未调用模型" : "用量未知，账本保留预留";
      const method = reply.answer_generation_method === "report_review_checklist" ? "原论文核对清单" : reply.context_kind || "研究对话";
      card.append(node("p", method + " · " + usage, "chat-caption"));
    }
    byId("workflow-messages").append(card);
    byId("workflow-messages").scrollTop = byId("workflow-messages").scrollHeight;
  }
  function renderTranscript(messages) {
    byId("workflow-messages").replaceChildren();
    for (const item of messages || []) appendMessage(item.role, item.text);
    if (!messages?.length) appendMessage("assistant", welcome);
  }
  function stepText(step) {
    return ({ collecting: "步骤 1 · 确认研究对象", clarifying_scope: "步骤 2 · 确认癌症范围", awaiting_retrieval_confirmation: "步骤 3 · 等待确认检索", resolving_entities: "步骤 4 · 处理实体歧义", results_ready: "步骤 5 · 选择后续分析", report_ready: "步骤 6 · 追问当前报告" })[step] || "研究流程处理中";
  }
  function setControls() {
    const busy = state.sending || !!state.job || state.uploading;
    byId("workflow-input").disabled = !!busy;
    byId("workflow-send").disabled = busy || !state.session;
    byId("workflow-new").disabled = busy;
    byId("workflow-stop").classList.toggle("hidden", !state.job || state.paused);
    byId("workflow-resume").classList.toggle("hidden", !state.job || !state.paused);
    byId("workflow-send").textContent = busy ? "处理中…" : "发送";
  }
  function abstracts(result) {
    const seen = new Set();
    return (result?.evidence || []).filter((record) => {
      const src = record.source || {};
      if (record.evidence_type !== "publication" || src.source_database !== "PubMed" || !record.summary || (record.warnings || []).includes("abstract_not_available")) return false;
      const key = String(src.record_id || src.doi || "").trim().toLowerCase();
      if (!key || seen.has(key)) return false;
      seen.add(key); return true;
    }).length;
  }
  function renderVersions(snapshot) {
    const box = byId("workflow-result-versions"); box.replaceChildren();
    const results = snapshot?.results || [];
    if (results.length < 2) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    box.append(node("label", "研究版本："));
    const select = document.createElement("select"); select.setAttribute("aria-label", "选择检索结果版本");
    for (const result of results) {
      const option = node("option", "v" + result.version + " · " + result.query); option.value = result.result_id;
      if (result.result_id === snapshot.current_result?.result_id) option.selected = true;
      select.append(option);
    }
    select.addEventListener("change", async () => {
      try { state.snapshot = await api("/api/workflow/select-result", { session_id: state.session, result_id: select.value }); renderState(state.snapshot); byId("workflow-status").textContent = "已切换结果版本；后续回答绑定当前版本，历史对话仍保留。"; }
      catch (error) { byId("workflow-status").textContent = error.message; }
    });
    box.append(select);
  }
  function renderResolution(snapshot) {
    const box = byId("workflow-resolution"); box.replaceChildren();
    const resolution = snapshot?.entity_resolution;
    if (!resolution?.required) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    box.append(node("strong", "实体存在多个候选，请先确认后续如何处理。", "workflow-action-title"));
    box.append(node("p", "选择候选只表示你的分析关注点，不会把来源歧义改写为已解析，也不会删除其他候选或证据。"));
    const select = document.createElement("select"); select.setAttribute("aria-label", "选择关注候选实体");
    select.append(node("option", "选择一个基因或疾病候选")); select.options[0].value = "";
    for (const item of resolution.candidates || []) {
      if (!item.id) continue;
      const label = (item.entity_type === "gene" ? "基因" : "疾病") + " · " + item.name + (item.identifier ? " · " + item.namespace + ":" + item.identifier : "");
      const option = node("option", label); option.value = item.id; select.append(option);
    }
    const actions = node("div", null, "workflow-action-buttons");
    const choose = node("button", "将候选作为关注点", "primary-button"); choose.type = "button"; choose.disabled = select.options.length < 2;
    choose.addEventListener("click", () => { if (select.value) void resolveEntity("select_candidate", select.value); });
    const keep = node("button", "保留歧义继续", "quiet-button"); keep.type = "button"; keep.addEventListener("click", () => void resolveEntity("keep_ambiguous"));
    actions.append(select, choose, keep); box.append(actions);
  }
  async function resolveEntity(decision, entityId) {
    try {
      const response = await api("/api/workflow/resolve-entity", {
        session_id: state.session, result_id: state.snapshot.current_result.result_id,
        decision, ...(entityId ? { entity_id: entityId } : {})
      });
      if (response.assistant_message) appendMessage("assistant", response.assistant_message);
      renderState(response);
      byId("workflow-status").textContent = "歧义处理已记录；原始实体候选与证据仍完整保留。";
    } catch (error) { byId("workflow-status").textContent = error.message; }
  }
  function renderAction(action) {
    const card = byId("workflow-action-card"); card.replaceChildren();
    if (!action) { card.classList.add("hidden"); return; }
    card.classList.remove("hidden");
    const title = names[action.kind] || "研究操作";
    const statusLabel = action.status === "pending" ? "等待确认"
      : action.status === "running" ? "执行中"
        : action.status === "completed" && action.kind === "localize" ? "已结束 · 请查看整理状态"
          : action.status === "completed" ? "已完成"
            : action.status === "failed" ? "未完成"
              : action.status === "cancelled" ? "已取消" : "条件已变更";
    card.append(node("strong", title + " · " + statusLabel, "workflow-action-title"));
    if (action.kind === "retrieve" && action.status === "pending") {
      const inputGrid = node("div", null, "workflow-action-grid");
      const gene = document.createElement("input"); gene.value = action.params.gene || ""; gene.maxLength = 120; gene.setAttribute("aria-label", "基因");
      const disease = document.createElement("input"); disease.value = action.params.disease || ""; disease.maxLength = 120; disease.setAttribute("aria-label", "英文疾病检索词"); disease.placeholder = "英文疾病检索词";
      inputGrid.append(gene, disease); card.append(inputGrid);
      if (action.params.disease_input && action.params.disease_input !== action.params.disease) {
        card.append(node("p", `中文疾病名“${action.params.disease_input}”已转换为英文检索词“${action.params.disease}”。请核对后再确认。`, "chat-caption"));
      }
      const sources = node("fieldset", null, "workflow-source-options"); sources.append(node("legend", "数据来源"));
      for (const [key, label] of [["pubmed", "PubMed"], ["uniprot", "UniProt"], ["opentargets", "Open Targets"]]) {
        const wrapper = node("label"); const check = document.createElement("input"); check.type = "checkbox"; check.value = key; check.checked = (action.params.sources || []).includes(key);
        wrapper.append(check, document.createTextNode(" " + label)); sources.append(wrapper);
      }
      const drug = document.createElement("input"); drug.type = "checkbox"; drug.checked = action.params.include_drugs === true;
      const drugLabel = node("label", null, "workflow-drug-option"); drugLabel.append(drug, document.createTextNode(" 包含疾病层级药物候选（非基因靶向药结论）"));
      const openTargets = [...sources.querySelectorAll('input[value="opentargets"]')][0];
      const syncDrug = () => { drug.disabled = !openTargets.checked; if (drug.disabled) drug.checked = false; };
      openTargets.addEventListener("change", syncDrug); syncDrug();
      card.append(sources, drugLabel, node("p", "只有点击确认后才查询。修改任何条件后，服务端会换发确认 ID 并要求再次确认。", "chat-caption"));
      const controls = node("div", null, "workflow-action-buttons");
      const confirm = node("button", "确认并检索", "primary-button"); confirm.type = "button";
      confirm.addEventListener("click", () => void confirmAction(action.action_id, "confirm", {
        gene: gene.value.trim(), disease: disease.value.trim(),
        sources: [...sources.querySelectorAll("input:checked")].map((item) => item.value), include_drugs: drug.checked
      }));
      const cancel = node("button", "取消", "quiet-button"); cancel.type = "button"; cancel.addEventListener("click", () => void confirmAction(action.action_id, "cancel"));
      controls.append(confirm, cancel); card.append(controls);
    } else if (action.status === "pending") {
      card.append(node("p", action.kind === "localize" ? "将仅整理当前结果版本，英文来源记录保持不变。" : "只使用本次结果中有摘要的 PubMed 论文；不跨检索版本混合。"));
      const controls = node("div", null, "workflow-action-buttons");
      const confirm = node("button", "确认执行", "primary-button"); confirm.type = "button"; confirm.addEventListener("click", () => void confirmAction(action.action_id, "confirm"));
      const cancel = node("button", "取消", "quiet-button"); cancel.type = "button"; cancel.addEventListener("click", () => void confirmAction(action.action_id, "cancel"));
      controls.append(confirm, cancel); card.append(controls);
    } else if (action.status === "running") {
      card.append(node("p", "状态由服务端返回；当前没有逐来源或阶段进度可供展示。"));
    } else if (action.status === "failed") {
      card.append(node("p", errors[action.error_code] || "操作未完成，原始证据仍保留。"));
    }
  }
  function renderNextActions(snapshot) {
    const holder = byId("workflow-next-actions");
    const result = snapshot?.current_result?.payload;
    if (!result || !["results_ready", "report_ready"].includes(snapshot.step)) { holder.classList.add("hidden"); return; }
    holder.classList.remove("hidden");
    const synthesisButton = holder.querySelector('[data-workflow-prepare="synthesize"]');
    synthesisButton.disabled = abstracts(result) < 2 || !!snapshot.pending_action;
    synthesisButton.title = abstracts(result) < 2 ? "至少需要两篇不同论文的摘要" : snapshot.pending_action ? "请先处理上一个待确认操作" : "准备综合确认卡";
  }
  function renderState(snapshot) {
    const contextKey = `${snapshot.current_result?.result_id || ""}:${snapshot.current_result?.synthesis_id || ""}`;
    if (state.attachmentContext !== undefined && state.attachmentContext !== contextKey)
      attachments?.reset("已切换证据上下文；如需引用文件，请重新添加。");
    state.attachmentContext = contextKey;
    state.snapshot = snapshot;
    byId("workflow-step").textContent = stepText(snapshot.step);
    renderAction(snapshot.pending_action || snapshot.last_action);
    renderVersions(snapshot);
    renderResolution(snapshot);
    renderNextActions(snapshot);
    const current = snapshot.current_result;
    if (current?.payload && current.result_id !== state.renderedResultId) {
      state.renderedResultId = current.result_id;
      if (window.geneEvidenceRenderResult) window.geneEvidenceRenderResult(current.gene || snapshot.gene || "未知基因", current.disease || snapshot.disease || "未知疾病", current.payload);
      if (current.synthesis_id && current.payload.result_id) {
        const entry = snapshot.results.find((item) => item.result_id === current.result_id);
        if (entry) entry.synthesis_id = current.synthesis_id;
      }
    }
  }
  async function refresh() {
    if (!state.session) return;
    state.snapshot = await api("/api/workflow/state?session_id=" + encodeURIComponent(state.session));
    renderState(state.snapshot);
  }
  async function confirmAction(actionId, decision, values = {}) {
    try {
      const data = { session_id: state.session, action_id: actionId, decision, ...values };
      const response = await api("/api/workflow/confirm", data);
      renderAction(response.action);
      if (response.assistant_message) appendMessage("assistant", response.assistant_message);
      if (response.action?.status === "running") {
        state.job = { kind: "action", action_id: response.action.action_id }; state.paused = false; setControls(); void pollAction();
      } else {
        byId("workflow-status").textContent = response.assistant_message || "操作状态已更新。";
        await refresh();
      }
    } catch (error) { byId("workflow-status").textContent = error.message; }
  }
  async function prepare(kind) {
    if (kind === "localize") {
      byId("workflow-status").textContent = "中文整理已移除；请查看原始英文证据。";
      return;
    }
    try {
      const response = await api("/api/workflow/prepare", { session_id: state.session, action: kind });
      appendMessage("assistant", response.assistant_message); renderAction(response.action);
      const mockMode = state.snapshot?.current_result?.payload?.mode === "mock";
      byId("workflow-status").textContent = mockMode
        ? "模拟确认仅演示流程，不会调用模型；英文原始证据保持不变。"
        : "需要单独确认后才会调用模型。";
    } catch (error) { byId("workflow-status").textContent = error.message; }
  }
  async function pollAction() {
    if (state.polling || !state.job) return;
    state.polling = true; setControls();
    try {
      while (state.job && !state.paused) {
        const snapshot = await api("/api/workflow/state?session_id=" + encodeURIComponent(state.session));
        renderState(snapshot);
        const action = snapshot.last_action;
        if (action?.action_id === state.job.action_id && ["completed", "failed", "cancelled"].includes(action.status)) {
          if (action.assistant_message) appendMessage("assistant", action.assistant_message);
          if (action.kind === "synthesize" && action.result?.synthesis) {
            window.dispatchEvent(new CustomEvent("gene-evidence:synthesis-result", { detail: { report: action.result.synthesis, query: snapshot.current_result?.query || "当前检索" } }));
          }
          byId("workflow-status").textContent = action.status === "completed"
            ? action.kind === "localize" ? "整理请求已结束；请核对右侧整理状态，原始证据已保留。" : "操作完成；结果版本和引用已保留。"
            : "操作未完成；没有自动重试。";
          state.job = null; break;
        }
        await new Promise((resolve) => setTimeout(resolve, 650));
      }
    } catch (error) {
      state.paused = true; byId("workflow-status").textContent = error.message + " 可继续查看同一任务，不会重新提交。";
    } finally { state.polling = false; setControls(); }
  }
  async function pollChat(job) {
    state.job = { kind: "chat", job_id: job.job_id }; setControls();
    try {
      while (state.job && !state.paused) {
        const current = await api("/api/workflow/chat/job?session_id=" + encodeURIComponent(state.session) + "&job_id=" + encodeURIComponent(job.job_id));
        if (current.reply) {
          if (current.status === "completed") appendMessage("assistant", current.reply.answer, current.reply);
          const reply = current.reply;
          if (current.status === "failed") {
            const usage = reply.usage?.status === "reported" ? `已记录 ${reply.usage.total_tokens} tokens。`
              : reply.usage?.status === "unknown" ? "用量未知，账本保留预留。" : "未调用模型。";
            const diagnostics = (reply.model_output_diagnostics || []).slice(0, 5)
              .map((item) => `${item.path}: ${item.error_type}`).join("；");
            byId("workflow-status").textContent = (errors[reply.error_code] || "研究对话失败；没有自动重试。")
              + " " + usage + (diagnostics ? " 安全校验诊断：" + diagnostics : "");
          } else {
            byId("workflow-status").textContent = "研究回答完成。下一步：" + stepText(state.snapshot?.step || "collecting");
          }
          state.job = null; break;
        }
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    } catch (error) { state.paused = true; byId("workflow-status").textContent = error.message + " 可继续读取同一回答。"; }
    finally { setControls(); }
  }
  async function send() {
    const text = byId("workflow-input").value.trim();
    if (!text || state.sending || state.job || !state.session || state.uploading) return;
    state.sending = true; setControls(); appendMessage("user", text); setComposerText("");
    const sources = [...document.querySelectorAll(".source-checkbox:checked")].map((input) => input.value);
    const includeDrugs = byId("include-drugs").checked && sources.includes("opentargets");
    try {
      const response = await api("/api/workflow/message", {
        session_id: state.session, message_id: crypto.randomUUID(), text, sources, include_drugs: includeDrugs,
        attachment_ids: attachments?.ids() || []
      });
      if (response.assistant_message) appendMessage("assistant", response.assistant_message);
      if (response.action) renderAction(response.action);
      byId("workflow-status").textContent = response.action?.status === "pending" ? "检索尚未执行，请检查确认卡。" : "";
      if (response.chat_job) await pollChat(response.chat_job);
      await refresh();
    } catch (error) { byId("workflow-status").textContent = error.message; }
    finally { state.sending = false; setControls(); }
  }
  async function startSession() {
    state.sending = true; setControls();
    try {
      const remembered = sessionStorage.getItem(sessionKey);
      if (remembered) {
        try { state.session = remembered; state.snapshot = await api("/api/workflow/state?session_id=" + encodeURIComponent(state.session)); }
        catch { state.session = null; }
      }
      if (!state.session) {
        const created = await api("/api/workflow/session", {}); state.session = created.session_id;
        sessionStorage.setItem(sessionKey, state.session); state.snapshot = created;
      }
      renderTranscript(state.snapshot.messages); renderState(state.snapshot);
      const health = await api("/api/health");
      const wf = health.guided_workflow || {};
      byId("workflow-mode").textContent = (health.mode === "mock" ? "模拟模式 · 不访问数据库或模型" : "本机工作流 · 检索按所选来源执行") + (wf.model_interpretation_enabled ? " · 模型辅助理解已启用" : " · 未启用解析模型，输入请尽量写成‘基因与癌症’") + " · 每项行动均需你确认";
      if (state.snapshot.current_result?.payload && window.geneEvidenceRenderResult) {
        const item = state.snapshot.current_result; state.renderedResultId = item.result_id;
        window.geneEvidenceRenderResult(item.gene || state.snapshot.gene || "未知基因", item.disease || state.snapshot.disease || "未知疾病", item.payload);
      }
    } catch (error) {
      byId("workflow-mode").textContent = "本机工作流暂不可用；请确认网页服务正在运行。";
      byId("workflow-status").textContent = error.message;
      renderTranscript([{ role: "assistant", text: welcome }]);
    } finally { state.sending = false; setControls(); }
  }
  byId("workflow-send").addEventListener("click", () => void send());
  byId("workflow-new").addEventListener("click", async () => {
    if (state.job || state.sending || state.uploading) return;
    attachments?.reset();
    sessionStorage.removeItem(sessionKey); state.session = null; state.snapshot = null; state.renderedResultId = null;
    setComposerText("");
    byId("workflow-action-card").classList.add("hidden"); byId("workflow-next-actions").classList.add("hidden");
    byId("workflow-resolution").classList.add("hidden");
    byId("results-view").classList.add("hidden"); byId("search-view").classList.remove("hidden");
    await startSession();
  });
  byId("chat-toggle").addEventListener("click", () => {
    const open = byId("chat-toggle").getAttribute("aria-expanded") !== "true";
    setChatOpen(open, open);
  });
  document.addEventListener("gea:education-view-hidden", () => setChatOpen(true, false));
  byId("chat-close").addEventListener("click", () => { setChatOpen(false); byId("chat-toggle").focus(); });
  byId("chat-backdrop").addEventListener("click", () => setChatOpen(false));
  byId("workflow-stop").addEventListener("click", () => { state.paused = true; byId("workflow-status").textContent = "已停止等待；后台执行继续结算。恢复时仍读取同一任务，不重复提交。"; setControls(); });
  byId("workflow-resume").addEventListener("click", () => { state.paused = false; if (state.job?.kind === "chat") void pollChat(state.job); else void pollAction(); });
  byId("workflow-messages").addEventListener("click", (event) => {
    const source = event.target.closest("a"); if (source) { source.target = "_blank"; source.rel = "noopener noreferrer"; }
  });
  document.querySelectorAll("[data-workflow-prepare]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.workflowPrepare === "synthesize") {
      window.dispatchEvent(new CustomEvent("gene-evidence:start-synthesis"));
    } else {
      void prepare(button.dataset.workflowPrepare);
    }
  }));
  window.geneEvidenceGuidedQuery = (gene, disease) => {
    setComposerText("请检索 " + gene + " 与 " + disease);
    setChatOpen(true);
    void send();
  };
  byId("workflow-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); void send(); }
  });
  byId("workflow-input").addEventListener("input", () => {
    byId("workflow-char-count").textContent = byId("workflow-input").value.length + "/3000";
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && window.matchMedia("(max-width: 1050px)").matches && document.body.classList.contains("chat-open")) {
      setChatOpen(false); byId("chat-toggle").focus();
    }
  });
  setChatOpen(true, false);
  if (window.matchMedia("(max-width: 1050px)").matches) setChatOpen(true, false);
  window.addEventListener("gene-evidence:quick-query", (event) => window.geneEvidenceGuidedQuery(event.detail.gene, event.detail.disease));
  window.addEventListener("gene-evidence:prepare-action", (event) => void prepare(event.detail.kind));
  window.addEventListener("gene-evidence:synthesis-followup", async (event) => {
    if (!state.session || state.job) { byId("workflow-status").textContent = "请等待当前操作完成后再切换报告。"; return; }
    try {
      state.snapshot = await api("/api/workflow/select-result", { session_id: state.session, result_id: event.detail.result_id });
      renderState(state.snapshot); setComposerText("这份报告最重要的证据缺口是什么？");
      setChatOpen(true, true);
      byId("workflow-status").textContent = "追问将绑定所选报告和其原始摘要；可连续提问。";
    } catch (error) { byId("workflow-status").textContent = error.message; }
  });
  window.addEventListener("gene-evidence:synthesis-notice", (event) => {
    if (event.detail?.text) appendMessage("assistant", event.detail.text);
  });
  window.addEventListener("gene-evidence:synthesis-completed", async (event) => {
    const resultId = event.detail?.result_id, synthesisId = event.detail?.synthesis_id;
    if (!state.session || !resultId || !synthesisId) return;
    try {
      const bound = await api("/api/workflow/synthesis-bind", {
        session_id: state.session, result_id: resultId, synthesis_id: synthesisId
      });
      if (bound.assistant_message) appendMessage("assistant", bound.assistant_message);
      renderState(bound);
    } catch (error) {
      byId("workflow-status").textContent = "报告已生成，但绑定到对话失败（" + error.message + "）；原报告仍可查看。";
    }
  });
  startSession();
})();
