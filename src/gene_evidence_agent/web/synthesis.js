/* Guided Plan-and-Solve flow. Every model stage requires its own server action ID. */
(() => {
  const n = (id) => document.getElementById(id);
  const state = {
    latest: null, job: null, paused: false, polling: false, enabled: false,
    followup: false, reports: new Map(), announced: new Set()
  };
  const labels = {
    awaiting_plan_start: "本地预检完成，等待你确认是否生成分析计划",
    planning: "正在根据原始英文摘要制定分析计划",
    awaiting_plan_review: "分析计划已生成，等待你审核或修改",
    extracting: "正在逐篇提取英文研究卡片",
    awaiting_study_review: "研究卡片已完成原句核验，等待你确认",
    synthesizing: "正在生成英文跨文献分析",
    awaiting_translation_decision: "英文报告已完成",
    translating: "正在生成报告",
    completed: "英文跨文献分析已完成",
    partial: "流程未全部完成；已保留已完成阶段",
    failed: "当前阶段失败；原始检索证据和此前完成阶段已保留",
    insufficient_evidence: "跨文献证据不足：至少需要两篇不同论文的英文摘要",
    mock: "模拟模式：未调用模型，不代表真实文献结论",
    cancelled: "已取消；此前检索证据仍保留"
  };
  const errorLabels = {
    insufficient_reported_findings: "不足两篇论文具有可引用的明确发现；已保留研究卡，未发起综合模型请求。请换选其他论文或调整计划。",
    output_token_limit_exceeded: "服务商报告的输出用量超过本次上限；结果未接受，已返回用量仍计入账本。",
    unsupported_numeric_claim: "有一条综合主张包含来源未支持的数字；该版本未发布。"
  };
  const reasons = {
    no_abstract: "无摘要", duplicate: "重复论文", invalid_record: "记录字段不完整",
    language_not_reported: "PubMed 未提供摘要语言元数据", non_english_abstract: "摘要语言不是英文",
    paper_limit: "超过最多 10 篇限制", input_budget: "完整摘要超过输入预算"
  };
  const types = {
    common_finding: "共同发现 · Common finding",
    difference: "差异 · Difference",
    conflict: "冲突 · Conflict",
    inference: "综合推断 · Inference",
    hypothesis: "研究假设 · Hypothesis (not validated)"
  };
  const warningLabels = {
    unsupported_absence_claim_excluded: "不受来源支持的“未开展/未验证”主张已逐条排除，其他有效主张保留；排除记录见报告。",
    incomparable_conflict_claim_excluded: "有一条模型生成的冲突主张因研究背景不可比或信息不足而被过滤；其他通过校验的主张仍保留。",
    unsupported_conflict_claim_excluded: "有一条模型生成的冲突主张缺少两篇论文各自明确发现的支持，已过滤；其他通过校验的主张仍保留。",
    insufficient_findings_support_claim_excluded: "有一条或多条跨篇主张没有引用至少两篇不同论文的发现，已过滤；其他通过校验的主张仍保留。",
    invalid_fact_reference_claim_excluded: "有一条或多条主张使用了无效或不匹配的事实引用，已过滤；保留内容仍只基于可追溯来源。",
    causal_overstatement_claim_excluded: "有一条结论把来源中的相关性或不确定表述升级成因果结论，已过滤；其他通过校验的主张仍保留。",
    causal_language_mismatch_flagged: "报告中有主张使用了强于来源措辞的因果表达；该主张已保留但标记了风险，不能据此认定因果关系。",
    unsupported_source_term_flagged: "有主张包含原始摘要中未检出的英文大写术语或缩写；该主张已保留并醒目标记，请核对它是否为来源术语的等价写法或无依据的新实体。",
    unsupported_numeric_claim_excluded: "有一条跨文献主张包含摘要未支持的数字，已过滤；其他通过校验的主张仍保留。",
    structured_comparison_only: "旧版逐篇对照报告"
  };
  const claimRiskLabels = {
    causal_language_mismatch: "风险提示：此项主张使用因果措辞，但引用来源含相关性或不确定表述；不能据此认定因果关系。",
    unsupported_source_term: "风险提示：此项主张含原始摘要中未检出的英文大写术语或缩写；需核对是否与来源术语等价，或是否为来源未支持的新实体。"
  };
  const active = new Set(["planning", "extracting", "synthesizing", "translating"]);
  const terminal = new Set(["completed", "partial", "failed", "insufficient_evidence", "mock", "cancelled"]);

  async function api(path, data) {
    const response = await fetch(path, data === undefined ? { cache: "no-store" } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data)
    });
    const value = await response.json();
    if (!response.ok) {
      const error = new Error(value.error || "api_unavailable");
      error.code = value.error;
      throw error;
    }
    return value;
  }
  function text(tag, content, className) {
    const node = document.createElement(tag);
    node.textContent = content == null ? "" : String(content);
    if (className) node.className = className;
    return node;
  }
  function notice(key, message) {
    if (state.announced.has(key)) return;
    state.announced.add(key);
    window.dispatchEvent(new CustomEvent("gene-evidence:synthesis-notice", { detail: { text: message } }));
  }
  function controls() {
    n("synthesis-generate").disabled = !!state.job || !state.enabled || !state.latest?.result_id
      || state.latest.publication_count < 2;
    n("synthesis-stop").classList.toggle("hidden", !state.job || state.paused || !active.has(state.job.status));
    n("synthesis-resume").classList.toggle("hidden", !state.job || !state.paused);
  }
  function references(parent, report, ids) {
    for (const id of [...new Set(ids || [])]) {
      const paper = report.papers.find((item) => item.evidence_id === id);
      if (!paper) continue;
      try {
        const url = new URL(paper.url);
        if (url.protocol !== "https:" || url.username || url.password) continue;
        const link = text("a", "PMID " + paper.pmid);
        link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer"; link.className = "source-link";
        parent.append(link, document.createTextNode(" "));
        if (report.result_id === state.latest?.result_id) {
          const button = text("button", "定位证据", "back-link"); button.type = "button";
          button.addEventListener("click", () => document.getElementById("chat-evidence-" + paper.record_index)
            ?.scrollIntoView({ behavior: "smooth", block: "center" }));
          parent.append(button);
        }
      } catch { /* Never render unsafe links. */ }
    }
  }
  function versions(selected = "") {
    const select = n("synthesis-version"); select.replaceChildren();
    const empty = text("option", "选择已生成的报告"); empty.value = ""; select.append(empty);
    for (const [id, entry] of state.reports) {
      const option = text("option", entry.query + " · " + id.slice(0, 8)); option.value = id; select.append(option);
    }
    select.value = selected;
  }
  function factText(fact) { return fact?.text_en || fact?.text_zh || "Not reported"; }
  function renderReport(report) {
    if (!report) return;
    const body = n("synthesis-body"); body.replaceChildren();
    if (report.report_markdown_en) {
      const download = text("button", "下载英文报告 (.md)", "quiet-button"); download.type = "button";
      download.addEventListener("click", () => {
        const url = URL.createObjectURL(new Blob([report.report_markdown_en], { type: "text/markdown;charset=utf-8" }));
        const link = document.createElement("a"); link.href = url;
        link.download = `gene-cancer-report-${report.synthesis_id}.md`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
      body.append(download);
    }
    body.append(text("p", "Cross-literature analysis · English · 仅以原始英文 PubMed 摘要和已核验英文事实为依据；未使用中文整理结果。"));
    body.append(text("p", "合格英文摘要 " + report.papers.length + " / 候选 " + report.candidate_count
      + " 篇；本报告按已确认计划选用 " + report.studies.length + " 篇。报告仅覆盖本次已获取摘要。", "chat-caption"));
    if (report.plan) {
      body.append(text("h4", "Approved analysis plan"));
      body.append(text("p", "Comparison dimensions: " + report.plan.comparison_dimensions_zh.join("; "), "synthesis-original"));
      body.append(text("p", "Questions: " + report.plan.comparison_questions_zh.join("; "), "synthesis-original"));
      if (report.plan.scope_limitations_zh?.length)
        body.append(text("p", "Scope limits: " + report.plan.scope_limitations_zh.join("; "), "chat-caption"));
    }
    if (report.excluded?.length)
      body.append(text("p", "排除记录：" + report.excluded.map((item) =>
        (reasons[item.reason] || item.reason) + "（记录 " + (item.record_index + 1) + "）").join("；")));
    if (report.studies?.length) {
      body.append(text("h4", "Per-paper study cards · English"));
      const wrap = text("div", "", "synthesis-table-wrap");
      const table = document.createElement("table"); table.className = "synthesis-table";
      const header = document.createElement("tr");
      for (const name of ["Paper / source", "Design and study context", "Scope and endpoint", "Findings and limitations"])
        header.append(text("th", name));
      table.append(header);
      for (const study of report.studies) {
        const paper = report.papers.find((item) => item.evidence_id === study.evidence_id);
        if (!paper) continue;
        const row = document.createElement("tr");
        const first = document.createElement("td");
        first.append(text("strong", paper.title), text("p", [paper.journal, paper.publication_date].filter(Boolean).join(" · ")));
        references(first, report, [study.evidence_id]);
        const context = (names) => names.map((name) => name + ": " + factText(study[name])).join("\n");
        first.append(text("p", study.evidence_role === "review" ? "Review-reported observations; not independent experimental replication." : "Primary study or unspecified record type.", "chat-caption"));
        row.append(first,
          text("td", context(["design", "population", "species", "cohort"]), "synthesis-original"),
          text("td", context(["cancer_subtype", "gene_variant", "endpoint"]), "synthesis-original"));
        const findings = document.createElement("td");
        for (const fact of study.findings || []) {
          findings.append(text("p", factText(fact), "synthesis-original"));
          if (fact.quote_en) findings.append(text("blockquote", fact.quote_en, "synthesis-quote"));
        }
        for (const fact of study.limitations || []) {
          findings.append(text("p", "Reported limitation: " + factText(fact), "synthesis-original"));
          if (fact.quote_en) findings.append(text("blockquote", fact.quote_en, "synthesis-quote"));
        }
        const details = document.createElement("details");
        details.append(text("summary", "Open the full original abstract"));
        details.append(text("p", paper.abstract_en, "synthesis-original")); findings.append(details);
        row.append(findings); table.append(row);
      }
      wrap.append(table); body.append(wrap);
    }
    for (const note of report.independence_notes || []) body.append(text("p", note, "chat-caption"));
    for (const warning of report.warnings || []) {
      if (warning === "structured_comparison_only") continue;
      body.append(text("p", warningLabels[warning] || "综合报告存在一项校验提示：" + warning, "chat-caption"));
    }
    if (report.analysis) {
      for (const item of report.analysis.excluded_claims || [])
        body.append(text("p", `Generated claim ${item.position} excluded: ${item.reason}. It is not an accepted finding.`, "chat-caption"));
      body.append(text("h4", "Cross-paper findings and interpretation · English"));
      body.append(text("p", "Model-generated analysis · each claim is linked to verified source facts. Review the original abstracts before relying on an interpretation; this is not a full-text or systematic review.", "chat-caption"));
      const claims = report.analysis.claims || [];
      for (const kind of ["common_finding", "difference", "conflict", "inference", "hypothesis"]) {
        const selected = claims.filter((claim) => claim.kind === kind);
        if (!selected.length) continue;
        body.append(text("h5", types[kind] || kind));
        for (const claim of selected) {
          const card = text("article", "", "synthesis-claim" + (claim.kind === "hypothesis" ? " synthesis-hypothesis" : ""));
          card.append(text("p", claim.text_en || ""));
          for (const risk of claim.risk_flags || []) {
            card.append(text("p", claimRiskLabels[risk] || "风险提示：该主张需要谨慎解读。", "synthesis-risk-flag"));
          }
          card.append(text("p", "Evidence basis: " + (claim.basis_en || "")));
          if (claim.hypothesis_status === "not_validated")
            card.append(text("p", "Hypothesis; not validated. " + (claim.verification_direction_en || "")));
          const evidenceIds = (claim.fact_refs || []).map((ref) => report.fact_catalog?.[ref]?.evidence_id).filter(Boolean);
          references(card, report, evidenceIds); body.append(card);
        }
      }
      if (!claims.length) body.append(text("p", "No cross-paper claims passed the evidence checks. The verified per-paper cards remain available above.", "chat-caption"));
      body.append(text("h5", "Evidence gaps · English"));
      for (const gap of report.analysis.evidence_gaps || []) {
        // Older cached reports may still contain the former verbose scope disclaimer.
        // The current scope is stated once above, so omit that legacy duplicate.
        if (gap.includes("This simplified report presents paper-by-paper facts side by side.")) continue;
        body.append(text("p", gap, "synthesis-original"));
      }
    }
    if (report.chinese_summary) {
      body.append(text("h4", "中文结论摘要 · 模型翻译"));
      for (const claim of report.chinese_summary.claims || []) {
        const card = text("article", "", "synthesis-claim" + (claim.kind === "hypothesis" ? " synthesis-hypothesis" : ""));
        card.append(text("strong", types[claim.kind] || claim.kind), text("p", claim.text_zh));
        for (const risk of claim.risk_flags || []) {
          card.append(text("p", claimRiskLabels[risk] || "风险提示：该主张需要谨慎解读。", "synthesis-risk-flag"));
        }
        if (claim.verification_direction_zh) card.append(text("p", "验证方向：" + claim.verification_direction_zh));
        const evidenceIds = (claim.fact_refs || []).map((ref) => report.fact_catalog?.[ref]?.evidence_id).filter(Boolean);
        references(card, report, evidenceIds); body.append(card);
      }
      body.append(text("h4", "中文证据缺口"));
      for (const gap of report.chinese_summary.evidence_gaps_zh || []) body.append(text("p", gap));
    }
    if (report.error_code) body.append(text("p", "阶段状态：" + report.error_code, "chat-caption"));
    const usage = (report.usage || []).map((item) => {
      const u = item.usage || {};
      return (item.stage || "stage") + ": " + (u.status === "reported" ? u.total_tokens + " tokens" : u.status);
    }).join(" · ");
    body.append(text("p", usage ? "分阶段用量：" + usage : "本报告没有可显示的已返回用量。", "chat-caption"));
    n("synthesis-followup").classList.toggle("hidden", !["completed", "mock"].includes(report.status));
    n("synthesis-followup").disabled = !state.followup;
  }
  function paperPicker(job, selectedIds, idPrefix) {
    const box = text("fieldset", "", "workflow-source-options");
    box.append(text("legend", "纳入论文（最多 10 篇；仅限服务端预检候选）"));
    for (const paper of job.papers || []) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox"; checkbox.value = paper.evidence_id;
      checkbox.checked = selectedIds.includes(paper.evidence_id); checkbox.dataset.paperChoice = idPrefix;
      label.append(checkbox, document.createTextNode(" " + paper.title + " · PMID " + paper.pmid));
      box.append(label, document.createElement("br"));
    }
    return box;
  }
  function selectedFrom(idPrefix) {
    return [...document.querySelectorAll('input[data-paper-choice="' + idPrefix + '"]:checked')].map((item) => item.value);
  }
  function lines(value) { return value.split(/\r?\n/).map((item) => item.trim()).filter(Boolean); }
  function actionButton(label, decision, extra = {}, className = "primary-button") {
    const button = text("button", label, className); button.type = "button";
    button.addEventListener("click", () => void confirmAction(decision, extra()));
    return button;
  }
  function renderReviewControls(job) {
    const body = n("synthesis-body"); const box = text("section", "", "workflow-action");
    const action = job.current_action;
    if (!action) return;
    const send = (decision, values = {}) => confirmAction(decision, values);
    if (job.status === "awaiting_plan_start") {
      box.append(text("strong", "第一步：确认是否生成分析计划"));
      box.append(text("p", "本地预检未调用模型。系统按 PMID/DOI 去重并排除无摘要记录；请选择论文并可补充分析重点。"));
      if (job.excluded?.length) box.append(text("p", "预检排除：" + job.excluded.map((item) => reasons[item.reason] || item.reason).join("；"), "chat-caption"));
      box.append(paperPicker(job, job.selected_evidence_ids || [], "start"));
      const focus = document.createElement("textarea"); focus.rows = 3; focus.maxLength = 3000;
      focus.placeholder = "可选：说明逐篇展示时重点关注哪些字段，例如：研究设计、癌症亚型和主要终点。";
      focus.setAttribute("aria-label", "补充分析重点"); box.append(focus);
      const buttons = text("div", "", "workflow-action-buttons");
      const go = text("button", "确认并生成分析计划", "primary-button"); go.type = "button";
      go.addEventListener("click", () => void send("confirm", {
        selected_evidence_ids: selectedFrom("start"),
        ...(lines(focus.value).length ? { focus_questions_zh: lines(focus.value) } : {})
      }));
      buttons.append(go, actionButton("取消", "cancel", () => ({}), "quiet-button")); box.append(buttons);
    } else if (job.status === "awaiting_plan_review" && job.plan) {
      box.append(text("strong", "第二步：审核或修改分析计划"));
      box.append(text("p", "以下计划由原始英文摘要生成。可调整纳入论文和分析重点；确认后才会调用模型逐篇提取英文事实。"));
      box.append(text("p", "比较维度：" + job.plan.comparison_dimensions_zh.join("；"), "synthesis-original"));
      box.append(text("p", "计划问题：" + job.plan.comparison_questions_zh.join("；"), "synthesis-original"));
      if (job.plan.scope_limitations_zh?.length)
        box.append(text("p", "计划范围限制：" + job.plan.scope_limitations_zh.join("；"), "chat-caption"));
      box.append(paperPicker(job, job.selected_evidence_ids || [], "plan"));
      const focus = document.createElement("textarea"); focus.rows = 4; focus.maxLength = 3000;
      focus.value = (job.focus_questions_zh || []).join("\n"); focus.setAttribute("aria-label", "已确认分析重点");
      box.append(text("label", "分析重点（每行一项）")); box.append(focus);
      const buttons = text("div", "", "workflow-action-buttons");
      const approve = text("button", "确认计划并提取英文研究卡", "primary-button"); approve.type = "button";
      approve.addEventListener("click", () => void send("confirm", {
        selected_evidence_ids: selectedFrom("plan"),
        ...(lines(focus.value).length ? { focus_questions_zh: lines(focus.value) } : {})
      }));
      buttons.append(approve, actionButton("取消", "cancel", () => ({}), "quiet-button")); box.append(buttons);
    } else if (job.status === "awaiting_study_review") {
      box.append(text("strong", "第三步：审核逐篇英文研究卡"));
      box.append(text("p", "每条已报告事实均附其原摘要中的英文原句；Not reported 表示摘要层面的卡片未提取到明确信息。"));
      const buttons = text("div", "", "workflow-action-buttons");
      const approve = text("button", "确认卡片并生成英文跨文献分析", "primary-button"); approve.type = "button";
      box.append(text("p", "确认后会单独调用一次模型，归纳共同发现、差异或冲突、有限推断及明确标为未验证的假设。论据与引用由服务器绑定到已核验事实；不生成中文综合译文。", "chat-caption"));
      approve.addEventListener("click", () => void send("confirm"));
      const revise = text("button", "返回修改论文或分析重点", "quiet-button"); revise.type = "button";
      revise.addEventListener("click", () => void send("revise_plan"));
      buttons.append(approve, revise, actionButton("取消", "cancel", () => ({}), "quiet-button")); box.append(buttons);
    }
    body.append(box);
  }
  function renderJob(job) {
    state.job = job;
    n("synthesis-status").textContent = (labels[job.status] || job.status)
      + (job.error_code ? "（" + (errorLabels[job.error_code] || job.error_code) + "）" : "");
    if (job.report) renderReport(job.report);
    else {
      const body = n("synthesis-body"); body.replaceChildren();
      body.append(text("p", "本地预检候选 " + job.candidate_count + " 篇；有摘要候选 " + job.papers.length + " 篇。分析计划、逐篇事实提取和最终英文综合均需分别确认；最终综合只读取原始英文摘要及已核验事实。"));
      if (job.excluded?.length)
        body.append(text("p", "未纳入：" + job.excluded.map((item) => (reasons[item.reason] || item.reason) + "（记录 " + (item.record_index + 1) + "）").join("；")));
      if (job.studies?.length) renderReport({ ...job, studies: job.studies, analysis: null, status: "partial", report_language: "en", usage: job.usage || [], fact_catalog: {} });
    }
    renderReviewControls(job);
    if (active.has(job.status)) {
      notice(job.job_id + ":" + job.status, ({
        planning: "我已开始分析规划；完成后会先展示计划，暂不会提取或综合。",
        extracting: "计划已确认。我正在逐篇提取原始英文摘要中的事实，完成后会等你审核研究卡。",
        synthesizing: "研究卡已确认。我正在根据原始英文摘要与已核验事实生成跨文献分析；完成后会展示共识、差异或冲突和待验证假设。",
        translating: "正在处理已确认的报告阶段。"
      })[job.status]);
    } else if (job.status === "awaiting_plan_review") {
      notice(job.job_id + ":plan", "分析计划已生成。请检查纳入论文和比较重点；确认前不会调用逐篇提取阶段。");
    } else if (job.status === "awaiting_study_review") {
      notice(job.job_id + ":cards", "逐篇英文研究卡已生成并完成原句校验。请审核后决定是否继续生成英文跨文献分析。");
    } else if (job.status === "awaiting_translation_decision") {
      notice(job.job_id + ":translation", "英文分析已完成。");
    }
    if (job.report?.synthesis_id) {
      state.reports.set(job.report.synthesis_id, { report: job.report, query: state.latest?.query || "当前检索" });
      versions(job.report.synthesis_id);
      if (job.report.status === "completed")
        window.dispatchEvent(new CustomEvent("gene-evidence:synthesis-completed", {
          detail: { result_id: job.result_id, synthesis_id: job.report.synthesis_id }
        }));
    }
    controls();
  }
  async function confirmAction(decision, values = {}) {
    const job = state.job; if (!job?.current_action) return;
    const request = {
      result_id: job.result_id, job_id: job.job_id, action_id: job.current_action.action_id,
      decision, ...values
    };
    try {
      const updated = await api("/api/synthesis/action", request);
      renderJob(updated);
      if (active.has(updated.status)) {
        state.paused = false; void poll();
      }
    } catch (error) {
      n("synthesis-status").textContent = "操作未提交（" + error.message + "）；没有自动重试。";
    }
  }
  async function poll() {
    if (state.polling || !state.job || !active.has(state.job.status)) return;
    state.polling = true; state.paused = false; controls();
    try {
      while (state.job && active.has(state.job.status) && !state.paused) {
        const job = await api("/api/synthesis/job?result_id=" + encodeURIComponent(state.job.result_id)
          + "&job_id=" + encodeURIComponent(state.job.job_id));
        renderJob(job);
        if (!active.has(job.status)) break;
        await new Promise((resolve) => setTimeout(resolve, 900));
      }
    } catch (error) {
      state.paused = true;
      n("synthesis-status").textContent = "读取任务状态失败（" + error.message + "）。继续查看会读取同一任务，不会重新提交模型调用。";
    } finally {
      state.polling = false; controls();
    }
  }
  async function start() {
    if (state.job || !state.latest?.result_id) return;
    if (!state.enabled) {
      n("synthesis-status").textContent = "当前服务器未启用逐篇文献对照服务；没有启动预检或模型请求。";
      return;
    }
    try {
      const job = await api("/api/synthesis", { result_id: state.latest.result_id });
      renderJob(job);
      if (active.has(job.status)) state.paused = false;
    } catch (error) {
      n("synthesis-status").textContent = "无法启动预检（" + error.message + "）。未自动重试。";
    }
  }
  async function configure() {
    try {
      const health = await api("/api/health");
      state.enabled = health.synthesis?.enabled === true;
      state.followup = health.chat?.synthesis_enabled === true;
      if (!state.enabled && state.latest?.result_id) {
        n("synthesis-status").textContent = "当前服务器未启用逐篇文献对照服务；原始检索证据仍可查看，未发起模型请求。";
      }
    } catch { state.enabled = false; state.followup = false; }
    controls();
  }
  window.addEventListener("gene-evidence:result", (event) => {
    state.latest = event.detail; state.job = null; state.announced.clear();
    n("synthesis-body").replaceChildren(); n("synthesis-followup").classList.add("hidden"); versions();
    n("synthesis-status").textContent = "先按原始英文摘要预检；计划、逐篇提取和跨文献分析均由你逐步确认。";
    controls(); void configure();
  });
  window.addEventListener("gene-evidence:synthesis-result", (event) => {
    const report = event.detail?.report; if (!report) return;
    state.reports.set(report.synthesis_id, { report, query: event.detail.query || "当前检索" });
    versions(report.synthesis_id); renderReport(report);
  });
  window.addEventListener("gene-evidence:start-synthesis", () => void start());
  n("synthesis-generate").addEventListener("click", () => void start());
  n("synthesis-stop").addEventListener("click", () => {
    state.paused = true;
    n("synthesis-status").textContent = "已暂停等待状态更新；后台当前阶段继续结算。恢复会读取同一任务，不会重发请求。";
    controls();
  });
  n("synthesis-resume").addEventListener("click", () => void poll());
  n("synthesis-version").addEventListener("change", () => {
    const entry = state.reports.get(n("synthesis-version").value);
    if (entry) renderReport(entry.report);
    else { n("synthesis-body").replaceChildren(); n("synthesis-followup").classList.add("hidden"); }
  });
  n("synthesis-followup").addEventListener("click", () => {
    const entry = state.reports.get(n("synthesis-version").value);
    if (entry) window.dispatchEvent(new CustomEvent("gene-evidence:synthesis-followup", {
      detail: { synthesis_id: entry.report.synthesis_id, result_id: entry.report.result_id, query: entry.query }
    }));
  });
  void configure();
})();
