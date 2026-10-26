// Local learning aids use the published catalog, not new model-generated medical facts.
(() => {
  const KEY = "gea.education.learning.v1";
  const validId = value => typeof value === "string" && /^K-edu-[a-z0-9-]{1,80}$/.test(value);
  const sections = new Set(["summary", "education-explanation", "education-faq-details", "education-media-details"]);
  const empty = () => ({ progress: null, notes: [] });
  function normalize(value) {
    if (!value || value.version !== 1) return empty();
    const progress = value.progress && validId(value.progress.entryId) ? {
      entryId: value.progress.entryId,
      section: sections.has(value.progress.section) ? value.progress.section : "summary",
    } : null;
    const notes = Array.isArray(value.notes) ? value.notes.filter(note => note && validId(note.entryId)
      && typeof note.id === "string" && /^[a-zA-Z0-9-]{1,80}$/.test(note.id)
      && typeof note.answer === "string" && typeof note.question === "string").slice(0, 20).map(note => ({
        id: note.id, entryId: note.entryId, title: String(note.title || "科普问答").slice(0, 160),
        question: note.question.slice(0, 1200), answer: note.answer.slice(0, 12000),
        date: String(note.date || "").slice(0, 40), mock: note.mock === true,
        citations: Array.isArray(note.citations) ? note.citations.filter(item => item && typeof item.url === "string")
          .slice(0, 20).map(item => ({ url: item.url.slice(0, 1000), title: String(item.title || "来源").slice(0, 500) })) : [],
      })) : [];
    return { progress, notes };
  }
  function load(storage) {
    try { return normalize(JSON.parse(storage?.getItem(KEY) || "null")); } catch { return empty(); }
  }
  function save(storage, value) {
    try { if (!storage) return false; storage.setItem(KEY, JSON.stringify({ version: 1, ...normalize({ version: 1, ...value }) })); return true; }
    catch { return false; }
  }
  function clear(storage) { try { if (!storage) return false; storage.removeItem(KEY); return true; } catch { return false; } }
  function matchedTerm(text, term) {
    const value = String(term || "").toLowerCase();
    if (value.length < 2) return false;
    const query = String(text || "").toLowerCase();
    if (/^[a-z0-9 _-]+$/.test(value)) {
      const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      return new RegExp(`(^|[^a-z0-9])${escaped}($|[^a-z0-9])`, "i").test(query);
    }
    return query.includes(value);
  }
  function matches(question, catalog) {
    const scored = catalog.filter(entry => entry.review_status === "source_checked").map(entry => {
      const terms = [entry.name, ...(entry.aliases || [])];
      const score = Math.max(0, ...terms.filter(term => matchedTerm(question, term)).map(term => term.length));
      return { entry, score };
    }).filter(item => item.score > 0).sort((a, b) => b.score - a.score || a.entry.id.localeCompare(b.entry.id));
    if (scored.length) return scored.slice(0, 4).map(item => item.entry);
    // Broad words only suggest entry points; the user, not this heuristic, confirms scope.
    const preference = /遗传|遗传性|基因/.test(question) ? "gene" : /癌|肿瘤/.test(question) ? "cancer" : "concept";
    return catalog.filter(entry => entry.review_status === "source_checked" && entry.kind === preference).slice(0, 3);
  }
  function suggestions(entry, question = "", answer = "") {
    if (!entry) return [];
    const faqs = (entry.faqs || []).map(faq => faq.question).filter(text => text !== question);
    const recent = `${question} ${answer}`;
    const terms = [...new Set(recent.match(/[a-zA-Z][a-zA-Z0-9-]{1,}|[\u4e00-\u9fff]{2}/g) || [])];
    const relevance = faq => terms.filter(term => matchedTerm(faq, term)).length;
    faqs.sort((a, b) => relevance(b) - relevance(a));
    return [...new Set(faqs)].slice(0, 3);
  }
  function glossary(text, catalog, excluding) {
    return catalog.filter(entry => entry.kind === "concept" && entry.id !== excluding
      && entry.review_status === "source_checked"
      && [entry.name, ...(entry.aliases || [])].some(term => matchedTerm(text, term))).slice(0, 6);
  }
  function quiz(entry, catalog) {
    const other = catalog.filter(item => item.id !== entry.id && item.kind === entry.kind && item.review_status === "source_checked").slice(0, 2);
    if (other.length < 2) return null;
    return { prompt: entry.summary, correctId: entry.id,
      options: [other[0], entry, other[1]].map(item => ({ id: item.id, name: item.name })) };
  }
  window.GeaEducationLearning = Object.freeze({ load, save, clear, matches, suggestions, glossary, quiz, matchedTerm });
})();
