const STORAGE_KEY = "gene-evidence-history-v1";
const byId = (id) => document.getElementById(id);

function translationScopeNotice(localized) {
  if (Number.isInteger(localized?.record_count) && localized.record_count > 0
      && localized.translated_record_count === localized.record_count) {
    return "全部返回记录均有通过校验的中文整理，但部分长文本只翻译了受长度限制的片段；完整英文原文仍保留。";
  }
  return "本次中文整理仅覆盖部分记录或长文本片段；未翻译的内容保留原始英文，请勿将其误作完整中文摘要。";
}
const sourceNames = { pubmed: "PubMed", uniprot: "UniProt", opentargets: "Open Targets" };
const statusNames = {
  not_requested: "未选择", success: "成功", empty: "无结果", ambiguous: "多个候选", partial_success: "部分成功",
  invalid_input: "输入无效", rate_limited: "受限", timeout: "超时", upstream_error: "来源错误",
};
function evidenceAvailabilityText(result, localized, labels) {
  if (result?.mode === "mock") return "检索状态：模拟数据，不代表真实来源结果。";
  const localizedLabel = localized?.evidence_availability_label_zh
    || labels?.[localized?.evidence_availability];
  if (localizedLabel) return `证据状态：${localizedLabel}`;

  const outcomes = Array.isArray(result?.source_outcomes)
    ? result.source_outcomes.filter((item) => item && item.status !== "not_requested")
    : [];
  const evidenceCount = Array.isArray(result?.evidence) ? result.evidence.length : null;
  const sourceRecordCount = outcomes.some((item) => Number(item.record_count) > 0);
  const ambiguous = outcomes.some((item) => item.status === "ambiguous");
  const incomplete = outcomes.some((item) => ["partial_success", "invalid_input", "rate_limited", "timeout", "upstream_error"].includes(item.status));

  if ((evidenceCount ?? 0) > 0 || sourceRecordCount) {
    const countText = evidenceCount === null ? "已返回证据记录" : `已返回 ${evidenceCount} 条证据记录`;
    const caveats = [];
    if (ambiguous) caveats.push("至少一个来源存在候选歧义");
    if (incomplete) caveats.push("至少一个来源未完整返回");
    const caveatText = caveats.length ? `；${caveats.join("，")}` : "";
    return `证据状态：${countText}${caveatText}；有记录不等于证据充分或因果关系成立。`;
  }

  if (outcomes.length && outcomes.every((item) => ["success", "empty"].includes(item.status))) {
    return "证据状态：来源检索已完成，但没有返回证据记录；不能据此断言不存在关联。";
  }
  if (!outcomes.length) return "证据状态：尚未执行来源检索。";
  return "证据状态：来源检索未完整或存在候选歧义，不能据此判断是否有证据；请查看各数据源状态。";
}
const sourceErrorNames = {
  invalid_input: "输入不符合该数据源要求",
  rate_limited: "数据源请求受到限流",
  timeout: "数据源请求超时",
  upstream_error: "数据源服务或响应异常",
};
const resolutionNames = {
  resolved: "已解析", ambiguous: "存在多个候选，待消歧", unresolved: "尚未解析",
};
const evidenceGroups = [
  ["论文", "publication"],
  ["蛋白注释", "protein_annotation"],
  ["靶点—疾病关联", "target_disease_association"],
  ["疾病相关药物候选", "disease_drug_candidate"],
];
const safeRecordHosts = new Set([
  "pubmed.ncbi.nlm.nih.gov", "www.uniprot.org", "rest.uniprot.org", "platform.opentargets.org",
]);

function safeRecordUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" && safeRecordHosts.has(parsed.hostname) ? parsed.href : null;
  } catch { return null; }
}

function addDetailRow(list, label, value) {
  if (value === undefined || value === null || value === "") return;
  const row = document.createElement("div");
  row.className = "provenance-row";
  const term = document.createElement("span");
  term.className = "provenance-label";
  term.textContent = label;
  const description = document.createElement("span");
  description.className = "provenance-value";
  description.textContent = Array.isArray(value) ? value.join(", ") : String(value);
  row.append(term, description);
  list.append(row);
}

function evidenceCard(record, localizedRecord = null, translationStatus = null) {
  const card = document.createElement("article");
  card.className = "record-card";
  const title = document.createElement("h4");
  const translated = localizedRecord?.translation_applied === true;
  const translationFallback = !translated && ["failed_fallback", "partial_output_limit", "partial_validation"].includes(translationStatus);
  title.textContent = (translated && localizedRecord.title_zh) || record.title || "未提供标题";
  const summaryLabel = document.createElement("div");
  summaryLabel.className = "record-source-label";
  summaryLabel.textContent = translated
    ? `${localizedRecord.translation_generation_method === "structured_source" ? "结构化来源直译 · 非模型生成" : "模型整理"} · ${localizedRecord.relation_label_zh} · ${localizedRecord.research_level_label_zh}`
    : translationFallback
      ? translationStatus === "partial_validation" ? "未展示通过校验的中文译文 · 保留英文原文" : "中文整理未完成 · 以下保留英文原文"
      : "来源英文内容 · 未经模型改写";
  const summary = document.createElement("p");
  summary.className = "record-summary";
  summary.textContent = (translated && localizedRecord.summary_zh) || record.summary || "来源未提供描述文本。";
  card.append(title, summaryLabel, summary);

  if (!translated && localizedRecord?.translation_diagnostic_code) {
    const rejectedNotice = document.createElement("div");
    rejectedNotice.className = "record-truncation-warning";
    rejectedNotice.textContent = `本条译文未通过校验：${localizedRecord.translation_diagnostic_zh || "内容完整性检查失败"}（${localizedRecord.translation_diagnostic_code}）；原始英文保留。证据 ID：${record.evidence_id || localizedRecord.evidence_id}`;
    card.append(rejectedNotice);
  }

  if (localizedRecord?.research_level_basis_label_zh) {
    const researchBasis = document.createElement("div");
    researchBasis.className = "record-research-basis";
    researchBasis.textContent = `研究层级判断依据（自动初筛）：${localizedRecord.research_level_basis_label_zh}（原始代码：${localizedRecord.research_level_basis}）`;
    card.append(researchBasis);
  }

  if (localizedRecord?.translation_truncated === true) {
    const truncationNotice = document.createElement("div");
    truncationNotice.className = "record-truncation-warning";
    truncationNotice.textContent = "本条中文整理只覆盖受长度限制的片段；请展开原始英文全文核对。";
    card.append(truncationNotice);
  }

  const readingLimits = [
    ...(Array.isArray(localizedRecord?.limitations_zh) ? localizedRecord.limitations_zh : []),
    ...(localizedRecord?.gene_drug_scope_zh ? [localizedRecord.gene_drug_scope_zh] : []),
  ].filter((item) => typeof item === "string" && item.trim());
  if (readingLimits.length) {
    const limits = document.createElement("div");
    limits.className = "record-limitations";
    const limitsLabel = document.createElement("strong");
    limitsLabel.textContent = "阅读提示与局限";
    const limitsList = document.createElement("ul");
    for (const item of readingLimits) {
      const line = document.createElement("li");
      line.textContent = item;
      limitsList.append(line);
    }
    limits.append(limitsLabel, limitsList);
    card.append(limits);
  }

  const sourceWarnings = Array.isArray(localizedRecord?.warnings_zh)
    ? localizedRecord.warnings_zh
    : [];
  const sourceWarningCodes = Array.isArray(localizedRecord?.warnings)
    ? localizedRecord.warnings
    : [];
  if (sourceWarnings.length) {
    const warnings = document.createElement("div");
    warnings.className = "record-source-warnings";
    const warningsLabel = document.createElement("strong");
    warningsLabel.textContent = "来源提示";
    const warningsList = document.createElement("ul");
    sourceWarnings.forEach((warning, index) => {
      const line = document.createElement("li");
      line.append(document.createTextNode(warning));
      if (sourceWarningCodes[index]) {
        const code = document.createElement("code");
        code.textContent = `（原始代码：${sourceWarningCodes[index]}）`;
        line.append(" ", code);
      }
      warningsList.append(line);
    });
    warnings.append(warningsLabel, warningsList);
    card.append(warnings);
  }

  const details = document.createElement("details");
  details.className = "provenance-details";
  const toggle = document.createElement("summary");
  toggle.textContent = "来源、编号与记录字段";
  const fields = document.createElement("div");
  fields.className = "provenance-fields";
  const source = record.source || {};
  addDetailRow(fields, "数据库", source.source_database);
  if (record.evidence_id) addDetailRow(fields, "Agent 证据 ID", record.evidence_id);
  if (translated) {
    addDetailRow(fields, "原始英文标题", localizedRecord.title_en);
    addDetailRow(fields, "原始英文摘要", localizedRecord.summary_en);
  }
  addDetailRow(fields, "记录 ID", source.record_id);
  addDetailRow(fields, "PMID", record.pmid);
  addDetailRow(fields, "DOI", record.doi);
  addDetailRow(fields, "期刊", record.journal);
  addDetailRow(fields, "日期", record.publication_date);
  addDetailRow(fields, "实体 ID", record.entity_ids || []);
  addDetailRow(fields, "检索时间", source.retrieved_at);
  const recordUrl = safeRecordUrl(source.record_url);
  if (recordUrl) {
    const row = document.createElement("div");
    row.className = "provenance-row";
    const label = document.createElement("span");
    label.className = "provenance-label";
    label.textContent = "原始来源";
    const link = document.createElement("a");
    link.className = "source-link";
    link.href = recordUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "在数据库中打开 ↗";
    row.append(label, link);
    fields.append(row);
  } else {
    addDetailRow(fields, "原始来源", "没有可验证的 HTTPS 来源链接");
  }
  const mirrors = record.also_reported_by || [];
  for (const reference of mirrors) {
    const row = document.createElement("div");
    row.className = "provenance-row";
    const label = document.createElement("span");
    label.className = "provenance-label";
    label.textContent = "其他来源";
    const value = document.createElement("span");
    value.className = "provenance-value";
    value.textContent = `${reference.source_database} · ${reference.record_id}`;
    const mirrorUrl = safeRecordUrl(reference.record_url);
    if (mirrorUrl) {
      const link = document.createElement("a");
      link.className = "source-link";
      link.href = mirrorUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "打开 ↗";
      value.append(" · ", link);
    }
    row.append(label, value);
    fields.append(row);
  }
  details.append(toggle, fields);
  card.append(details);
  return card;
}

function readHistory() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}

function saveHistory(gene, disease) {
  const items = [{ gene, disease, at: new Date().toISOString() }, ...readHistory()]
    .filter((item, index, all) => all.findIndex((other) => other.gene === item.gene && other.disease === item.disease) === index)
    .slice(0, 20);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  renderHistory();
}

function renderHistory() {
  const list = byId("history-list");
  const items = readHistory();
  if (!items.length) {
    list.innerHTML = '<p class="empty-history">还没有本机查询记录。</p>';
    return;
  }
  list.replaceChildren(...items.map((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    const query = document.createElement("span");
    query.className = "history-query";
    query.textContent = `${item.gene} 与 ${item.disease}`;
    const date = document.createElement("span");
    date.className = "history-date";
    date.textContent = new Date(item.at).toLocaleString();
    button.append(query, date);
    button.addEventListener("click", () => {
      byId("gene-input").value = item.gene;
      byId("disease-input").value = item.disease;
      closeHistory();
      returnToSearch();
    });
    return button;
  }));
}

function localMockResponse(gene, disease) {
  return {
    mode: "mock",
    status: "partial_success",
    query: `Gene ${gene}; disease ${disease}`,
    summary: "这是界面联调模拟信息，不是对基因、疾病或治疗效果的真实判断。当前没有真实来源记录。",
    source_outcomes: [
      { source: "pubmed", status: "success", record_count: 1 },
      { source: "uniprot", status: "ambiguous", record_count: 2 },
      { source: "opentargets", status: "success", record_count: 1 },
    ],
    entities: [
      { entity_type: "gene", query_name: gene, resolution_status: "unresolved" },
      { entity_type: "disease", query_name: disease, resolution_status: "unresolved" },
    ],
    evidence: [],
  };
}

function showResults(gene, disease, result) {
  window.dispatchEvent(new CustomEvent("gene-evidence:result", { detail: { result_id: result.result_id || null, query: `${gene} 与 ${disease}`, mode: result.mode,
    publication_count: (result.evidence || []).filter((r) => r.evidence_type === "publication" && r.source?.source_database === "PubMed" && !(r.warnings || []).includes("abstract_not_available")).length } }));
  byId("search-view").classList.add("hidden");
  byId("results-view").classList.remove("hidden");
  byId("result-title").replaceChildren(document.createTextNode(gene), document.createTextNode(" 与 "), document.createTextNode(disease));
  const localized = result.chinese_report || null;
  const translationLabels = {
    completed: "模型整理 · 批量中文摘要",
    partial_output_limit: "模型中文整理 · 片段或记录范围受限",
    partial_validation: "完整性校验未完全通过 · 仅展示通过项",
    failed_fallback: "模型整理失败 · 英文原文保留",
    model_disabled: "模型整理未启用 · 原始证据保留",
    not_needed_no_records: "没有返回记录 · 无需整理",
  };
  byId("summary-text").textContent = localized?.overview_zh || result.summary || (result.mode === "mock"
    ? "这是界面联调模拟信息，不是研究结论。"
    : "模型中文整理未请求；以下保留数据库原始记录，供你逐项核验。");
  const availabilityLabels = {
    evidence_found: "已返回证据记录；这不等于证据充分或已证明因果关系",
    partial_evidence: "已返回部分记录，但部分来源有歧义或不完整",
    no_records_returned: "检索完成但没有返回证据记录；不能据此断言不存在关联",
    retrieval_incomplete: "检索不完整，不能据此判断是否有证据",
    not_retrieved: "尚未执行足以判断的来源检索",
  };
  const adequacyLabels = {
    not_assessed: "尚未评估；有记录不代表证据充分或因果关系成立",
    model_marked_insufficient: "本次输入材料不足以形成结论主张；不代表证明不存在关联",
    no_validated_claims: "本次未生成通过校验的结论主张；不代表证明不存在关联",
    no_records_to_assess: "没有记录可评估；请结合检索状态区分无结果与检索不完整",
  };
  byId("evidence-availability-text").textContent = evidenceAvailabilityText(result, localized, availabilityLabels);
  byId("evidence-adequacy-text").textContent = result.mode === "mock"
    ? "证据充分性：模拟内容，不作判断。"
    : `证据充分性：${localized?.evidence_adequacy_label_zh || adequacyLabels[localized?.evidence_adequacy] || "未提供充分性评估；有记录不代表证据充分。"}`;
  const coverage = byId("translation-coverage-text");
  coverage.hidden = true;
  if (result.mode === "mock") {
    coverage.textContent = "中文翻译覆盖：模拟状态，不作判断。";
  } else if (
    localized
    && Number.isInteger(localized.translated_record_count)
    && Number.isInteger(localized.record_count)
    && localized.translated_record_count >= 0
    && localized.record_count >= localized.translated_record_count
  ) {
    const translatedCount = localized.translated_record_count;
    const recordCount = localized.record_count;
    const retainedNote = translatedCount < recordCount
      ? "未展示通过校验的中文译文仍保留英文原文。"
      : "";
    const selectedCount = localized.translation_selected_record_count;
    const scopeNote = Number.isInteger(selectedCount) && selectedCount >= translatedCount && selectedCount <= recordCount
      ? `本轮选取 ${selectedCount} 条，${translatedCount} 条有通过校验的中文；其余 ${recordCount - selectedCount} 条未选入本轮翻译范围。`
      : "";
    const structured=localized.structured_translated_record_count||0;
    const methodNote=structured ? ` 其中${structured}条为精确结构化来源直译，非模型翻译。` : "";
    coverage.textContent = `中文翻译覆盖：${translatedCount}/${recordCount} 条记录。${retainedNote}` + scopeNote + methodNote;
  } else {
    coverage.textContent = "中文翻译覆盖信息未提供；请查看原始英文记录。";
  }
  const fallbackNote = byId("localization-fallback-note");
  fallbackNote.hidden = true;
  const translationStatus = localized?.translation_status;
  const fallbackMessages = {
    failed_fallback: "本次中文整理未完成；没有把空白或无效译文当作结果，下面保留原始英文记录。",
    partial_output_limit: translationScopeNotice(localized),
    partial_validation: "本次部分输出未通过逐项完整性检查；只展示通过的译文/结论，其余模型内容已过滤，原始英文记录仍保留。",
  };
  const diagnosticMessages = {
    translation_ids_mismatch: "模型返回的证据编号与请求不一致",
    translation_title_presence_changed: "模型改变了原始标题字段状态",
    english_summary_not_translated: "摘要没有完成中文翻译",
    english_title_not_translated: "自然语言英文标题未完成翻译",
    english_claim_not_translated: "结论没有完成中文翻译",
    insufficiency_not_translated: "证据不足说明没有完成中文翻译",
    protected_literal_omitted: "译文遗漏了来源中的受保护标识或数字",
    machine_literal_introduced: "译文加入了来源中没有的数字或变异写法",
    latin_term_introduced: "译文加入了来源中没有的英文词或缩写",
    unknown_claim_evidence_id: "结论引用了不存在的证据编号",
    claim_content_not_in_source: "结论包含原文片段未支持的内容",
    claim_not_grounded_in_translation: "结论不是逐字摘自所引且已校验的中文译文",
    disease_drug_overclaim: "药物候选关系表述超过来源字段范围",
    noncausal_record_overstated: "非因果类记录被表述成因果结论",
    causal_wording_not_source_supported: "原文片段没有明确支持因果措辞",
    negative_association_not_source_supported: "译文新增了原文未明确表达的无关联结论",
  };
  const fallbackMessage = fallbackMessages[translationStatus];
  const diagnosticMessage = localized?.translation_diagnostic_zh
    || diagnosticMessages[localized?.translation_diagnostic_code]
    || "";
  const unmatchedLatinTermMessage = localized?.translation_diagnostic_code === "latin_term_introduced"
    && Number.isInteger(localized?.translation_unmatched_latin_term_count)
    ? ` 来源外英文词/缩写计数：${localized.translation_unmatched_latin_term_count}（仅保留数量，不保存具体词项）。`
    : "";
  const finishReasonMessage = localized?.translation_finish_reason
    ? ` 模型结束情况：${localized.translation_finish_reason_label_zh || "服务端返回了其他结束标记，需核对"}（原始标记：${localized.translation_finish_reason}）。`
    : "";
  const translationErrorMessage = localized?.translation_error_code
    ? ` 翻译问题：${localized.translation_error_label_zh || "中文整理返回未分类错误"}（${localized.translation_error_code}）。`
    : "";
  const reasoningContentMessage = localized?.translation_reasoning_content_present === true
    ? " 服务端响应另含独立推理字段；该字段内容不展示。"
    : "";
  const diagnosticSuffix = diagnosticMessage
    ? ` 校验提示：${diagnosticMessage}。${unmatchedLatinTermMessage}`
    : "";
  const fallbackText = fallbackMessage
    ? `${fallbackMessage}${translationErrorMessage}${diagnosticSuffix}${finishReasonMessage}${reasoningContentMessage}`
      + (localized?.translation_schema_diagnostics || []).map(d =>
        ` 结构诊断：${d.path} / ${d.error_type}${d.character_limit ? ` / 字符上限${d.character_limit}` : ""}。`).join("")
    : "";
  fallbackNote.classList.toggle("hidden", !fallbackText);
  fallbackNote.textContent = fallbackText;
  document.querySelector(".summary-card .card-kicker").textContent = result.mode === "mock"
    ? "模拟内容 · 非研究摘要"
    : localized?.translation_status === "partial_output_limit"
      ? translationLabels.partial_output_limit
      : localized?.translation_status_label_zh || translationLabels[localized?.translation_status] || "模型整理未请求 · 原始证据优先";
  const claimList = byId("model-claims");
  claimList.replaceChildren();
  if (localized?.claims?.length) {
    const label = document.createElement("div");
    label.className = "card-kicker";
    label.textContent = "引用证据要点 · 可追溯证据 ID";
    const note = document.createElement("p");
    note.className = "summary-status";
    note.textContent = localized?.claim_generation_method === "extractive_validated"
      ? localized.structured_translated_record_count ? "程序摘录：以下句子逐字来自已校验中文视图（模型译文或结构化来源直译），不是新的推断。" : "程序摘录：以下完整句子逐字摘自所引的已校验模型中文译文，不是跨文献综合结论或新的模型推断。"
      : "以下句子逐字摘自所引的已校验中文译文，不是跨文献综合结论。";
    claimList.append(label, note);
    for (const claim of localized.claims) {
      const item = document.createElement("div");
      item.className = "model-claim";
      const text = document.createElement("span");
      text.textContent = claim.text_zh;
      const refs = document.createElement("cite");
      refs.textContent = `（证据：${claim.evidence_ids.join(", ")}）`;
      item.append(text, " ", refs);
      claimList.append(item);
    }
  }
  const localizedEvidenceGaps = Array.isArray(localized?.evidence_gaps_zh)
    ? localized.evidence_gaps_zh
    : [];
  const rawEvidenceGaps = Array.isArray(localized?.evidence_gaps_en)
    ? localized.evidence_gaps_en
    : [];
  if (localizedEvidenceGaps.length || rawEvidenceGaps.length) {
    const gapBlock = document.createElement("section");
    gapBlock.className = "evidence-gaps";
    const gapLabel = document.createElement("strong");
    gapLabel.textContent = "检索缺口与来源问题";
    const gapNote = document.createElement("p");
    gapNote.textContent = "以下为报告记录的检索缺口；没有检索到记录不等于相关关系不存在。";
    gapBlock.append(gapLabel, gapNote);
    if (localizedEvidenceGaps.length) {
      const gapList = document.createElement("ul");
      for (const gap of localizedEvidenceGaps) {
        const item = document.createElement("li");
        item.textContent = gap;
        gapList.append(item);
      }
      gapBlock.append(gapList);
    }
    if (rawEvidenceGaps.length) {
      const rawLabel = document.createElement("strong");
      rawLabel.textContent = "报告原始缺口描述（英文）";
      const rawList = document.createElement("ul");
      for (const gap of rawEvidenceGaps) {
        const item = document.createElement("li");
        item.textContent = gap;
        rawList.append(item);
      }
      gapBlock.append(rawLabel, rawList);
    }
    claimList.append(gapBlock);
  }
  const conflictFindings = Array.isArray(localized?.conflicting_findings_en)
    ? localized.conflicting_findings_en
    : [];
  if (conflictFindings.length) {
    const conflictBlock = document.createElement("section");
    conflictBlock.className = "conflicting-findings";
    const conflictLabel = document.createElement("strong");
    conflictLabel.textContent = "原报告中的冲突描述（英文原文）";
    const conflictNote = document.createElement("p");
    conflictNote.textContent = localized.conflicting_findings_note_zh
      || "这些描述尚未由当前中文流程逐条核验；请查看引用来源。";
    const conflictList = document.createElement("ul");
    for (const finding of conflictFindings) {
      const item = document.createElement("li");
      item.textContent = finding;
      conflictList.append(item);
    }
    conflictBlock.append(conflictLabel, conflictNote, conflictList);
    claimList.append(conflictBlock);
  }
  const reportWarnings = Array.isArray(localized?.warnings_zh) ? localized.warnings_zh : [];
  const reportLimitations = Array.isArray(localized?.limitations_zh) ? localized.limitations_zh : [];
  const insufficiencyNote = typeof localized?.insufficient_evidence_zh === "string"
    ? localized.insufficient_evidence_zh
    : "";
  if (reportWarnings.length || reportLimitations.length || insufficiencyNote) {
    const notesBlock = document.createElement("section");
    notesBlock.className = "report-notes";
    const notesTitle = document.createElement("strong");
    notesTitle.textContent = "报告提示与局限";
    notesBlock.append(notesTitle);
    if (insufficiencyNote) {
      const insufficiency = document.createElement("p");
      insufficiency.className = "report-insufficiency";
      insufficiency.textContent = insufficiencyNote;
      notesBlock.append(insufficiency);
    }
    for (const [label, entries] of [["报告警告", reportWarnings], ["总体局限", reportLimitations]]) {
      if (!entries.length) continue;
      const group = document.createElement("div");
      group.className = "report-note-group";
      const groupTitle = document.createElement("span");
      groupTitle.textContent = label;
      const list = document.createElement("ul");
      for (const entry of entries) {
        const item = document.createElement("li");
        item.textContent = entry;
        list.append(item);
      }
      group.append(groupTitle, list);
      notesBlock.append(group);
    }
    claimList.append(notesBlock);
  }
  document.querySelector(".demo-label").textContent = result.mode === "mock" ? "模拟数据" : "真实检索";
  byId("source-status-caption").textContent = result.mode === "mock"
    ? "全部为模拟状态"
    : "状态以本次实时数据源结果为准";
  byId("demo-banner").textContent = result.mode === "mock"
    ? "本机模拟后端 · 当前没有访问外部数据库或模型"
    : "本机检索 · 查询词已发送至所选外部数据库";
  byId("result-caption").textContent = result.mode === "mock"
    ? "来源状态与证据分组如下。演示内容仅用于界面预览。"
    : "来源状态与证据分组如下；请通过原始来源核验每项发现。";

  const sourceGrid = byId("source-grid");
  sourceGrid.replaceChildren(...(result.source_outcomes || []).map((source) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const top = document.createElement("div");
    top.className = "source-top";
    const name = document.createElement("span");
    name.className = "source-name";
    name.textContent = sourceNames[source.source] || source.source;
    const status = document.createElement("span");
    status.className = `source-status ${source.status}`;
    const statusLabel = source.status === "empty" && source.source === "opentargets" && source.record_count > 0
      ? "暂无关联证据"
      : statusNames[source.status] || `未知来源状态（${source.status || "未提供"}）`;
    status.textContent = result.mode === "mock" ? `模拟${statusLabel}` : statusLabel;
    const detail = document.createElement("div");
    detail.className = "source-detail";
    const localizedOutcome = localized?.source_outcomes?.find((item) => item.source === source.source);
    const errorLabel = source.error_code
      ? localizedOutcome?.error_code_label_zh || sourceErrorNames[source.error_code] || "来源返回未分类错误"
      : "";
    detail.textContent = result.mode === "mock"
      ? "模拟响应；没有请求真实数据库"
      : `${source.record_count} 项候选/证据数据${source.error_code ? ` · ${errorLabel}（${source.error_code}）` : ""}`;
    top.append(name, status);
    card.append(top, detail);
    return card;
  }));

  const entityGrid = byId("entity-grid");
  entityGrid.replaceChildren(...[
    { entityType: "gene", label: "基因", query: gene },
    { entityType: "disease", label: "疾病", query: disease },
  ].map((entity) => {
    const candidates = (result.entities || []).filter((item) => item.entity_type === entity.entityType);
    const queryEntity = candidates.find((item) => item.id?.startsWith(`query:${entity.entityType}:`))
      || candidates.find((item) => item.query_name?.toLocaleLowerCase() === entity.query.toLocaleLowerCase());
    const displayName = queryEntity?.canonical_name || entity.query;
    const candidateMap = new Map();
    for (const candidate of candidates) {
      if (!candidate.id?.startsWith("opentargets:")) continue;
      const candidateKey = candidate.identifier && candidate.namespace
        ? `${candidate.namespace}:${candidate.identifier}`
        : candidate.id;
      if (candidateKey) candidateMap.set(candidateKey, candidate);
    }
    const choices = [...candidateMap.values()];
    const card = document.createElement("div");
    card.className = "entity-card";
    const label = document.createElement("div");
    label.className = "entity-label";
    const resolutionView = localized?.entity_resolution?.find(
      (item) => item.query_entity?.id === queryEntity?.id,
    );
    const resolutionStatus = resolutionView?.resolution_status_label_zh
      || resolutionNames[queryEntity?.resolution_status]
      || "状态未知";
    label.textContent = result.mode === "mock"
      ? `${entity.label} · 模拟${resolutionStatus}`
      : `${entity.label} · ${resolutionStatus}`;
    const value = document.createElement("div");
    value.className = "entity-value";
    value.textContent = displayName;
    const identifier = document.createElement("div");
    identifier.className = "entity-label";
    identifier.textContent = queryEntity?.identifier
      ? `首选标识符：${queryEntity.namespace}:${queryEntity.identifier}`
      : result.mode === "mock" ? "模拟记录没有真实标识符" : "尚未解析到唯一首选标识符";
    card.append(label, value, identifier);
    if (choices.length) {
      const list = document.createElement("div");
      list.className = "candidate-list";
      for (const candidate of choices) {
        const item = document.createElement("span");
        item.className = "candidate-chip";
        const identifierText = candidate.namespace && candidate.identifier
          ? `${candidate.namespace}:${candidate.identifier}`
          : `${candidate.id || "无来源标识"} · 未解析`;
        item.textContent = `${candidate.canonical_name || candidate.query_name} · ${identifierText}`;
        list.append(item);
      }
      card.append(list);
    }
    if (result.mode === "mock") {
      const warning = document.createElement("div");
      warning.className = "ambiguous-callout";
      warning.textContent = "模拟解析：没有真实候选结果。";
      card.append(warning);
    } else if (choices.length > 1) {
      const warning = document.createElement("div");
      warning.className = "ambiguous-callout";
      warning.textContent = queryEntity?.resolution_status === "resolved"
        ? "检索发现多个候选；首选项按唯一完全名称匹配，其他候选仍保留，请核对标识符。"
        : "发现多个候选，系统未自动选择；请核对名称与标识符，当前证据仍保留该歧义。";
      card.append(warning);
    }
    return card;
  }));

  const groups = byId("evidence-groups");
  const localizedById = new Map((localized?.evidence || []).map((record) => [record.evidence_id, record]));
  groups.replaceChildren(...evidenceGroups.map(([name, evidenceType]) => {
    const records = (result.evidence || []).filter((record) => record.evidence_type === evidenceType);
    const section = document.createElement("div");
    section.className = "evidence-group";
    section.dataset.evidenceType = evidenceType;
    const head = document.createElement("div");
    head.className = "group-head";
    const title = document.createElement("strong");
    title.textContent = name;
    const count = document.createElement("span");
    count.className = "group-count";
    count.textContent = result.mode === "mock" ? "模拟占位" : `${records.length} 条`;
    head.append(title, count);
    section.append(head);
    if (evidenceType === "disease_drug_candidate") {
      const caution = document.createElement("div");
      caution.className = "drug-caution";
      caution.textContent = "疾病层级药物候选，不是该基因的靶向药证明，也不代表疗效或个体适用性。";
      section.append(caution);
    }
    if (records.length) {
      const recordList = document.createElement("div");
      recordList.className = "record-list";
      recordList.replaceChildren(...records.map((record) => {
        const card = evidenceCard(record, record.evidence_id ? localizedById.get(record.evidence_id) : null, localized?.translation_status);
        card.id = `chat-evidence-${result.evidence.indexOf(record)}`;
        return card;
      }));
      section.append(recordList);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "group-placeholder";
      placeholder.textContent = result.mode === "mock" ? "模拟占位：没有真实记录、编号或来源链接。" : "本次没有该类证据记录。";
      section.append(placeholder);
    }
    return section;
  }));
  saveHistory(gene, disease);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function runQuery(gene, disease) {
  const button = byId("search-button");
  byId("api-error").classList.add("hidden");
  const sources = [...document.querySelectorAll(".source-checkbox:checked")].map((input) => input.value);
  if (!sources.length) {
    byId("api-error").textContent = "请至少选择一个数据来源。";
    byId("api-error").classList.remove("hidden");
    return;
  }
  const includeDrugs = byId("include-drugs").checked && sources.includes("opentargets");
  button.disabled = true;
  button.innerHTML = '查询中… <span aria-hidden="true">·</span>';
  let result;
  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gene, disease, sources, include_drugs: includeDrugs }),
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error || `query_failed_${response.status}`);
    }
    result = await response.json();
  } catch (error) {
    // Keeps the P1 static prototype previewable without the P2 local server.
    if (window.location.protocol === "file:") {
      result = localMockResponse(gene, disease);
    } else {
      byId("api-error").textContent = error?.message === "unsupported_chinese_disease_term"
        ? "这个中文疾病名称暂未收录可靠英文映射，尚未检索。请改填英文疾病名称后再试。"
        : error?.message === "english_gene_symbol_required"
          ? "请使用英文基因符号（例如 TP53）；当前没有启动检索。"
        : "本机接口暂不可用；没有把请求伪装成成功，也未进行自动重试。请检查本地服务后再试。";
      byId("api-error").classList.remove("hidden");
      return;
    }
  } finally {
    button.disabled = false;
    button.innerHTML = '开始检索 <span aria-hidden="true">→</span>';
  }
  showResults(gene, disease, result);
  void configureRuntime();
}

// The guided workflow uses this same renderer for the server-owned result version.
window.geneEvidenceRenderResult = showResults;
window.geneEvidenceReturnToSearch = returnToSearch;

async function configureRuntime() {
  const runtimeNote = byId("demo-banner");
  if (window.location.protocol === "file:") {
    runtimeNote.textContent = "原型预览 · 模拟数据";
    return;
  }
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health_unavailable");
    const health = await response.json();
    runtimeNote.textContent = health.mode === "live"
      ? "真实数据源模式 · 查询词会发送到所选外部数据库"
      : health.mode === "mock" ? "模拟模式 · 不访问数据库或模型"
        : "无法确认运行模式，请检查本机服务";
  } catch {
    runtimeNote.textContent = "本机服务暂不可用，请检查后再查询";
  }
}

function returnToSearch() {
  byId("results-view").classList.add("hidden");
  byId("search-view").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openHistory() {
  byId("history-drawer").classList.remove("hidden");
  byId("drawer-backdrop").classList.remove("hidden");
  byId("history-toggle").setAttribute("aria-expanded", "true");
  renderHistory();
}

function closeHistory() {
  byId("history-drawer").classList.add("hidden");
  byId("drawer-backdrop").classList.add("hidden");
  byId("history-toggle").setAttribute("aria-expanded", "false");
}

byId("query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const gene = byId("gene-input").value.trim();
  const disease = byId("disease-input").value.trim();
  if (gene && disease) {
    if (window.location.protocol !== "file:" && window.geneEvidenceGuidedQuery) {
      window.geneEvidenceGuidedQuery(gene, disease);
    } else {
      void runQuery(gene, disease);
    }
  }
});
document.querySelectorAll(".example-chip").forEach((button) => button.addEventListener("click", () => {
  byId("gene-input").value = button.dataset.gene;
  byId("disease-input").value = button.dataset.disease;
}));
function syncDrugOption() {
  const openTargetsSelected = [...document.querySelectorAll(".source-checkbox:checked")]
    .some((input) => input.value === "opentargets");
  byId("include-drugs").disabled = !openTargetsSelected;
  byId("include-drugs").closest(".drug-option").classList.toggle("disabled", !openTargetsSelected);
}
document.querySelectorAll(".source-checkbox").forEach((input) => input.addEventListener("change", syncDrugOption));
syncDrugOption();
void configureRuntime();
byId("back-button").addEventListener("click", returnToSearch);
byId("history-toggle").addEventListener("click", openHistory);
byId("history-close").addEventListener("click", closeHistory);
byId("drawer-backdrop").addEventListener("click", closeHistory);
byId("history-clear").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  renderHistory();
});
