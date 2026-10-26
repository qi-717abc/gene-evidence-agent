# Gene Evidence Agent

面向普通人的基因与癌症科普助手，附有可追溯的研究工作台。它可浏览基础知识卡、向模型提问，并从 PubMed、UniProt 和 Open Targets 查询公开证据。

研究结果会保留来源标识与链接；药物候选按疾病关联展示，不代表它们就是某个基因的靶向药。项目用于科普与资料检索，不能代替医生诊断或治疗建议。

## 本机启动

需要 Python 3.11 及以上版本和 [uv](https://docs.astral.sh/uv/)。

```powershell
uv sync --locked
Copy-Item .env.example .env
```

离线预览不需要模型密钥：

```powershell
.\start-mock.ps1
```

浏览器打开 <http://127.0.0.1:8765/>。

配置 OpenAI 兼容模型服务和 NCBI 联系邮箱后，可以启动科普问答：

```powershell
notepad .env
.\start-education.ps1
```

`start-full.ps1` 会额外启用研究对话和英文文献对照功能。模型密钥保存在本机 `.env`，不要提交到 GitHub。查询词会发送到相应的公开数据库；启用模型时，问题和所选证据会发送到配置的模型服务。请勿输入患者身份信息。

## 主要目录

- `src/gene_evidence_agent/`：Agent、网页、知识卡和运行所需资源。
- `start-*.ps1`：离线预览、科普模式和完整模式启动脚本。
- `LICENSES-AND-DATA-SOURCES.md`：数据来源及第三方许可说明。

模型与 PubMed 配置项见 `.env.example`。完整依赖版本由 `uv.lock` 固定。
