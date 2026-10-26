# License and data-source notes

Reviewed on 2026-09-29. This is an engineering inventory, not legal advice or
a guarantee that any particular deployment or redistribution is permitted.
Review the live terms and the rights for each record before commercial use,
bulk redistribution, or offering this project as a service.

## Project code and upstream agent projects

- `pyproject.toml` declares this project as Apache-2.0; `LICENSE` contains the
  standard license text. This is the license configured for this project, not
  a grant of rights over third-party data returned at runtime.
- Biomni's repository is Apache-2.0, but its own license notes warn that
  integrated tools, databases, and software can have separate and sometimes
  stricter terms. This implementation uses Biomni as a design reference; it
  does not import Biomni Python modules.
- The Bio Agent Skills catalog is not a runtime dependency of this project.
  A skill's license, cited software, and referenced datasets would need a
  separate review before copying or bundling any skill content.

## Vendored model tokenizer and encoder

- `src/gene_evidence_agent/vendor/deepseek_v4_0e1a0e5/` contains the official
  DeepSeek-V4-Pro `tokenizer.json` and `encoding_dsv4.py` from immutable
  upstream revision `0e1a0e5e52aea73055f50fef6f2423db370265b6`, plus the
  upstream MIT `LICENSE` and a provenance/checksum record. The assets are used
  only for local prompt token preflight; no model weights are downloaded or
  executed. Recheck upstream terms and notices before redistribution.
- Runtime package dependency `tokenizers==0.23.2` is constrained in
  `pyproject.toml` and locked in `uv.lock`; PyPI lists it under Apache-2.0.
  Its transitive dependencies are also recorded in the lockfile.
- Official references: [DeepSeek-V4-Pro model and encoding files](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/0e1a0e5e52aea73055f50fef6f2423db370265b6), [DeepSeek V4 tokenizer provenance](https://github.com/deepseek-ai/deepseek-recipe/blob/main/static/tokenizers/v4/README.md), and [tokenizers on PyPI](https://pypi.org/project/tokenizers/0.23.2/).

## Runtime data sources

| Source | What this Agent requests | Licensing / operating notes |
|---|---|---|
| NCBI PubMed E-utilities | PubMed IDs and article metadata, including abstracts when supplied | NCBI requests an identifying tool name/email and documents request-rate limits (3 requests/second without an API key; 10/second with one). This client requires the software developer's contact email and paces requests accordingly. PubMed is an index; abstracts or full text may remain subject to publisher/author copyright. Show the PubMed citation/link and do not bulk republish abstracts without checking record-level rights. |
| UniProtKB REST API | Human gene/protein identity and available FUNCTION/DISEASE comments | UniProt data use Creative Commons Attribution 4.0 for copyrightable database content; retain attribution and source links when displaying or redistributing data. Verify current API usage and record-specific notices before bulk use. |
| Open Targets Platform GraphQL API | Target/disease entity matches, target-disease association evidence, and disease drug candidates | Open Targets describes Platform data as CC0 1.0 and its code as Apache-2.0, but it aggregates upstream data whose source-specific terms may differ. Keep provenance and original source references; check the particular contributor data before redistribution or commercial use. |
| SDU-hosted OpenAI-compatible model API | User query plus bounded retrieved evidence for planning/summarization when explicitly enabled | The provider's current service contract, data retention/training policy, allowed data, pricing, and billing behavior were not independently verified. Do not send patient or confidential records. Confirm these terms with the provider before production use. |

## Curated education catalog

- `src/gene_evidence_agent/education/catalog.json` contains concise Chinese paraphrases and links for introductory cancer, gene, and terminology cards. Cancer pages primarily link to the U.S. National Cancer Institute (NCI); gene cards link to NCBI Gene; the classification-concepts card links to WHO ICD-O. The catalog does not mirror full articles, PubMed abstracts, or images.
- NCI states that its information may be reused subject to its reuse guidance and source credit; use the original page title/link and check any page-specific notices before redistributing. NCBI Gene pages remain source records; this catalog links rather than republishes their complete records. Do not interpret this engineering note as legal clearance for bulk reuse or commercial distribution.
- The application's `source_checked` flag means a short entry has an identified source link and was checked against source material. It does not mean independent expert review. The catalog groups cancer topics by organ/system for navigation, not as a complete WHO/IARC histopathological classification.
- Official references: [NCI A to Z cancer types](https://www.cancer.gov/types), [NCI cancer types and cell/organ naming](https://www.cancer.gov/about-cancer/understanding/what-is-cancer), [NCI reuse and copyright](https://www.cancer.gov/policies/copyright-reuse), [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene), and [WHO International Classification of Diseases for Oncology](https://www.who.int/standards/classifications/other-classifications/international-classification-of-diseases-for-oncology).

## Operational limits and attribution

- The current adapters use fixed public API endpoints; the model cannot choose
  an endpoint or execute generated code.
- This Agent is a research-evidence retrieval aid, not a clinical decision
  system. A reported association, score, or drug development stage is not proof
  of causality, efficacy, approval, or suitability for an individual.
- Preserve source names, stable identifiers, citations, and outbound links in
  downstream reports. This does not replace source-specific attribution or
  permission requirements.
- Before a public or commercial deployment, maintain a source-level inventory
  for the exact dataset fields and records exposed, refresh upstream terms,
  document model-provider data handling, and obtain legal review as needed.

## Official references

- [Apache License 2.0 text](https://www.apache.org/licenses/LICENSE-2.0.txt)
- [Biomni README](https://github.com/snap-stanford/Biomni/blob/main/README.md)
- [Biomni third-party license notes](https://github.com/snap-stanford/Biomni/blob/main/license_info.md)
- [NCBI E-utilities guide](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
- [NLM/PubMed/PMC disclaimer](https://pmc.ncbi.nlm.nih.gov/about/disclaimer/)
- [UniProt API support and data licensing](https://www.uniprot.org/api-documentation/support-data)
- [Open Targets licensing](https://platform-docs.opentargets.org/licence)
- [Open Targets API documentation](https://platform-docs.opentargets.org/data-access/graphql-api)
