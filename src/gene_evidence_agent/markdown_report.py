"""Human-readable Chinese rendering of the validated evidence view."""

from __future__ import annotations

import re
from urllib.parse import quote

from gene_evidence_agent.chinese_report import (
    AGENT_RUN_STATUS_LABELS_ZH,
    ENTITY_RESOLUTION_LABELS_ZH,
    EVIDENCE_ADEQUACY_LABELS_ZH,
    EVIDENCE_AVAILABILITY_LABELS_ZH,
    SOURCE_ERROR_CODE_LABELS_ZH,
    TRANSLATION_DIAGNOSTIC_LABELS_ZH,
    TRANSLATION_ERROR_LABELS_ZH,
    TRANSLATION_FINISH_REASON_LABELS_ZH,
    TRANSLATION_STATUS_LABELS_ZH,
    UNKNOWN_RESEARCH_LEVEL_BASIS_ZH,
    ChineseEvidenceReport,
    LocalizedEvidenceItem,
    agent_run_warning_label_zh,
)
from gene_evidence_agent.schemas import BiomedicalEntity

_ENTITY_TYPE_LABELS = {
    "gene": "基因",
    "disease": "疾病",
    "drug": "药物",
    "variant": "变异",
    "protein": "蛋白质",
}
def render_chinese_markdown(
    report: ChineseEvidenceReport,
    *,
    run_status: str | None = None,
    run_warnings: list[str] | None = None,
    clarification_question: str | None = None,
) -> str:
    """Render a complete reading view without changing or omitting evidence fields."""

    lines = [
        "# 基因—疾病—药物证据报告",
        "",
        f"**查询：** {_md(report.query)}",
        f"**检索概况：** {_md(report.overview_zh)}",
        f"**证据状态：** {_md(report.evidence_availability_label_zh or EVIDENCE_AVAILABILITY_LABELS_ZH[report.evidence_availability])}",
        f"**证据充分性：** {_md(report.evidence_adequacy_label_zh or EVIDENCE_ADEQUACY_LABELS_ZH[report.evidence_adequacy])}",
        f"**中文处理：** {_md(report.translation_status_label_zh or TRANSLATION_STATUS_LABELS_ZH[report.translation_status])}",
        f"**记录数：** {report.record_count}；**已翻译：** {report.translated_record_count}",
    ]
    if report.structured_translated_record_count:
        lines.append(f"**中文来源：** {report.structured_translated_record_count}条精确结构化来源直译，非模型生成。")
    if run_status:
        status_label = AGENT_RUN_STATUS_LABELS_ZH.get(run_status, f"运行状态未知（{run_status}）")
        lines.append(f"**Agent 状态：** {_md(status_label)}")
    if report.translation_error_code:
        error_label = report.translation_error_label_zh or TRANSLATION_ERROR_LABELS_ZH.get(
            report.translation_error_code, "中文整理返回未分类错误"
        )
        lines.append(
            f"**翻译处理问题：** {_md(error_label)}（`{_code(report.translation_error_code)}`）"
        )
    if report.translation_diagnostic_code:
        diagnostic = report.translation_diagnostic_code
        label = report.translation_diagnostic_zh or TRANSLATION_DIAGNOSTIC_LABELS_ZH.get(diagnostic)
        detail = f"{_md(label)}（`{_code(diagnostic)}`）" if label else f"`{_code(diagnostic)}`"
        if report.translation_unmatched_latin_term_count is not None:
            detail += (
                f"；来源外英文词/缩写计数：{report.translation_unmatched_latin_term_count}"
                "（仅保留数量，不保存具体词项）"
            )
        lines.append(f"**安全校验提示：** {detail}")
    if report.translation_finish_reason:
        finish_reason = report.translation_finish_reason
        label = report.translation_finish_reason_label_zh or TRANSLATION_FINISH_REASON_LABELS_ZH.get(
            finish_reason.casefold(), "服务端返回了其他结束标记，需核对"
        )
        lines.append(f"**模型结束状态：** {_md(label)}（`{_code(finish_reason)}`）")
    if report.translation_reasoning_content_present is True:
        lines.append("**模型响应说明：** 服务端返回了独立推理字段；为避免展示内部推理文本，仅记录字段存在，不输出其内容。")
    if clarification_question:
        lines.extend(["", "## 需要补充的信息", "", _quote(clarification_question)])

    lines.extend(["", "## 数据源状态", ""])
    if report.source_outcomes:
        for outcome in report.source_outcomes:
            error = ""
            if outcome.error_code:
                error_label = outcome.error_code_label_zh or SOURCE_ERROR_CODE_LABELS_ZH.get(
                    outcome.error_code, "来源返回未分类错误"
                )
                error = f"；错误说明：{_md(error_label)}（`{_code(outcome.error_code)}`）"
            lines.append(
                f"- **{_md(outcome.source_label_zh)}：** {_md(outcome.status_label_zh)} "
                f"（{outcome.record_count} 条记录；原始状态 `{_code(outcome.status)}`{error}）"
            )
    else:
        lines.append("- 当前报告没有来源状态记录。")

    lines.extend(["", "## 实体消歧", ""])
    if report.entity_resolution:
        for resolution in report.entity_resolution:
            query = resolution.query_entity
            lines.append(
                f"- **查询{_ENTITY_TYPE_LABELS[query.entity_type]}：** "
                f"{_md(query.canonical_name or query.query_name)}；"
                f"状态：{ENTITY_RESOLUTION_LABELS_ZH[query.resolution_status]}"
            )
            lines.append(f"  - {_md(resolution.note_zh)}")
            if query.identifier:
                lines.append(
                    f"  - 标识符：`{_code(query.identifier)}`（{_code(query.namespace or '')}）"
                )
            if resolution.candidate_entities:
                lines.append("  - 候选项：")
                for candidate in resolution.candidate_entities:
                    lines.append(f"    - {_entity_line(candidate)}")
    else:
        lines.append("- 没有单独的实体消歧结果。")

    lines.extend(["", "## 报告中的实体", ""])
    if report.entities:
        for entity in report.entities:
            lines.append(f"- {_entity_line(entity)}")
    else:
        lines.append("- 未返回实体记录。")

    lines.extend(["", "## 引用证据要点", ""])
    if report.claims:
        lines.extend(
            [
                (
                    "程序摘录：以下完整句子逐字摘自所引的已校验中文译文，不是跨文献综合结论或新的模型推断。"
                    if report.claim_generation_method == "extractive_validated"
                    else "以下句子逐字摘自所引的已校验中文译文，不是跨文献综合结论。"
                ),
                "",
            ]
        )
        for index, claim in enumerate(report.claims, start=1):
            citations = ", ".join(_evidence_link(item) for item in claim.evidence_ids)
            lines.append(f"{index}. {_md(claim.text_zh)}（证据：{citations}）")
    elif report.insufficient_evidence_zh:
        lines.append(_md(report.insufficient_evidence_zh))
    else:
        lines.append("未生成中文结论；请查看下方逐条证据及原文。")

    if report.conflicting_findings_en:
        lines.extend(
            [
                "",
                "## 原报告中的冲突描述（英文原文）",
                "",
                f"**核验说明：** {_md(report.conflicting_findings_note_zh or '这些描述尚未由当前中文流程逐条核验。')}",
            ]
        )
        for finding in report.conflicting_findings_en:
            lines.extend(["", _quote(finding)])

    lines.extend(["", "## 按关系分组的证据", ""])
    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    rendered_ids: set[str] = set()
    if report.groups:
        for group in report.groups:
            lines.extend(
                [
                    f"### {_md(group.label_zh)}（{len(group.evidence_ids)} 条）",
                    "",
                    _md(group.interpretation_zh),
                ]
            )
            if group.gene_drug_conclusion_zh:
                lines.extend(["", f"**药物关系边界：** {_md(group.gene_drug_conclusion_zh)}"])
            for evidence_id in group.evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item is not None:
                    lines.extend(["", *_render_evidence(item)])
                    rendered_ids.add(evidence_id)
    for item in report.evidence:
        if item.evidence_id not in rendered_ids:
            lines.extend(["", f"### {_md(item.relation_label_zh)}", "", *_render_evidence(item)])

    if report.evidence_gaps_zh or report.evidence_gaps_en or report.warnings_zh or run_warnings or report.limitations_zh:
        lines.extend(["", "## 检索缺口、警告与局限", ""])
        for label, entries in (
            ("证据缺口", report.evidence_gaps_zh),
            ("警告", report.warnings_zh),
            ("Agent 警告", run_warnings or []),
            ("局限", report.limitations_zh),
        ):
            for entry in entries:
                if label == "Agent 警告":
                    explanation = agent_run_warning_label_zh(entry)
                    lines.append(
                        f"- **{label}：** {_md(explanation)}（原始代码 `{_code(entry)}`）"
                    )
                else:
                    lines.append(f"- **{label}：** {_md(entry)}")
        for entry in report.evidence_gaps_en:
            lines.append(f"- **报告原始缺口描述（英文）：** {_md(entry)}")

    lines.extend(
        [
            "",
            "---",
            "本报告用于整理研究证据，不构成诊断或个人治疗建议。中文翻译与自动分类仍需结合原文和专业领域审查。",
            "",
        ]
    )
    return "\n".join(lines)


def render_chinese_status_markdown(
    *,
    run_status: str,
    run_warnings: list[str],
    clarification_question: str | None = None,
) -> str:
    """Render a Chinese status page when planning stops before evidence retrieval."""

    status_label = AGENT_RUN_STATUS_LABELS_ZH.get(run_status, f"运行状态未知（{run_status}）")
    lines = [
        "# Agent 运行结果",
        "",
        f"**运行状态：** {_md(status_label)}",
    ]
    if clarification_question:
        lines.extend(
            [
                "",
                "## 模型返回的澄清问题（原文）",
                "",
                _quote(clarification_question),
            ]
        )
    if run_warnings:
        lines.extend(["", "## 警告", ""])
        lines.extend(
            f"- {_md(agent_run_warning_label_zh(item))}（原始代码 `{_code(item)}`）"
            for item in run_warnings
        )
    lines.extend(["", "本次未生成证据报告。", ""])
    return "\n".join(lines)


def _render_evidence(item: LocalizedEvidenceItem) -> list[str]:
    anchor = re.sub(r"[^A-Za-z0-9_-]", "", item.evidence_id)
    title = item.title_zh or ("中文标题未生成" if item.title_en else "来源未提供标题")
    lines = [f'<a id="evidence-{anchor}"></a>', f"#### {_md(title)}", ""]
    lines.append(f"- **证据 ID：** `{_code(item.evidence_id)}`")
    lines.append(
        f"- **类型/研究层级：** {_md(item.evidence_type_label_zh)}；"
        f"{_md(item.research_level_label_zh)}"
    )
    level_basis = item.research_level_basis_label_zh or UNKNOWN_RESEARCH_LEVEL_BASIS_ZH
    lines.append(
        f"- **研究层级判断依据（自动初筛）：** {_md(level_basis)} "
        f"（原始依据 `{_code(item.research_level_basis)}`）"
    )
    lines.append(f"- **分组：** {_md(item.relation_label_zh)}")
    lines.append(
        f"- **来源：** {_md(item.source.source_database)}；记录号 `{_code(item.source.record_id)}`；"
        f"[打开来源记录]({_markdown_url(str(item.source.record_url))})"
    )
    if item.pmid:
        lines.append(f"- **PMID：** `{_code(item.pmid)}`")
    if item.doi:
        lines.append(f"- **DOI：** `{_code(item.doi)}`")
    if item.score is not None:
        lines.append(f"- **数据库分数：** {item.score:.4g}（不是因果概率）")
    if item.entity_ids:
        lines.append("- **关联实体 ID：** " + ", ".join(f"`{_code(value)}`" for value in item.entity_ids))
    if item.also_reported_by:
        sources = ", ".join(
            f"{_md(ref.source_database)} `{_code(ref.record_id)}` "
            f"[来源链接]({_markdown_url(str(ref.record_url))})"
            for ref in item.also_reported_by
        )
        lines.append(f"- **其他来源副本：** {sources}")
    lines.extend(["", "**中文摘要：**"])
    if item.translation_diagnostic_code:
        lines.extend([
            "",
            f"**翻译未通过校验：** {_md(item.translation_diagnostic_zh or '')} "
            f"（`{_code(item.translation_diagnostic_code)}`；证据 `{_code(item.evidence_id)}`）",
        ])
    if item.summary_zh:
        lines.extend(["", _quote(item.summary_zh)])
    else:
        lines.extend(["", "中文翻译未完成；下方保留英文原文摘要。"])
    if item.title_en:
        lines.extend(["", "**英文原标题：**", "", _quote(item.title_en)])
    lines.extend(["", "**英文原文摘要：**", "", _quote(item.summary_en)])
    if item.gene_drug_scope_zh:
        lines.extend(["", f"**药物关系限定：** {_md(item.gene_drug_scope_zh)}"])
    for limitation in item.limitations_zh:
        lines.append(f"- **解读限制：** {_md(limitation)}")
    for warning, warning_zh in zip(item.warnings, item.warnings_zh, strict=True):
        lines.append(
            f"- **来源提示：** {_md(warning_zh)}（原始代码 `{_code(warning)}`）"
        )
    return lines


def _entity_line(entity: BiomedicalEntity) -> str:
    label = _ENTITY_TYPE_LABELS.get(entity.entity_type, entity.entity_type)
    name = entity.canonical_name or entity.query_name
    parts = [f"**{label}：** {_md(name)}", f"状态：{ENTITY_RESOLUTION_LABELS_ZH[entity.resolution_status]}"]
    if entity.identifier:
        parts.append(f"标识符：`{_code(entity.identifier)}`（{_code(entity.namespace or '')}）")
    if entity.canonical_name and entity.canonical_name != entity.query_name:
        parts.append(f"查询名：{_md(entity.query_name)}")
    return "；".join(parts)


def _evidence_link(evidence_id: str) -> str:
    anchor = re.sub(r"[^A-Za-z0-9_-]", "", evidence_id)
    return f"[`{_code(evidence_id)}`](#evidence-{anchor})"


def _quote(value: str) -> str:
    return "\n".join(f"> {_md(line)}" for line in value.splitlines())


def _md(value: str) -> str:
    """Escape untrusted text before placing it in a Markdown paragraph or heading."""

    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"([\\`*_{}\[\]()#+!|>])", r"\\\1", cleaned).replace("\n", " ")


def _code(value: str) -> str:
    return value.replace("`", "\\`").replace("\r", " ").replace("\n", " ")


def _markdown_url(value: str) -> str:
    """Keep URL structure while encoding delimiters that can break Markdown links."""

    return quote(value, safe=":/?&=#%")
