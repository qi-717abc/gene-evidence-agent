/* Reading a file is local. Only sending a question starts a model request. */
(() => {
  const errors = {
    unsupported_file_type: "请转成 PDF、DOCX、XLSX、PPTX 或文本文件。",
    file_size_limit: "单个文件需小于 8 MB。", document_archive_limit: "文档解压内容过大或已加密。",
    no_readable_text: "未读到文字；扫描 PDF 和图片需先做文字识别。",
    document_read_failed: "无法读取，请检查文件或转成 PDF/文本。",
    encrypted_document: "请先移除文档密码。", document_page_limit: "PDF 最多读取 200 页。",
    attachment_capacity: "每个会话最多加入 4 个文件；更多文件请新建对话。",
    attachment_not_found: "文件已过期或不属于当前会话，请重新添加。",
    session_not_found: "会话已过期，请新建对话。", session_busy: "请等待当前回答完成。"
  };
  window.GeaAttachments = {
    create({ input, channel, getSession, canUpload = () => true, onBusy = () => {} }) {
      const host = document.getElementById(input);
      if (!host) return null;
      let items = [], uploading = false;
      const box = document.createElement("div"); box.className = "attachment-composer";
      const picker = document.createElement("input"); picker.type = "file"; picker.multiple = true;
      picker.accept = ".pdf,.docx,.xlsx,.pptx,.txt,.md,.csv,.tsv,.json,.html,.htm,.xml,.rtf,.odt,.ods,.odp";
      picker.hidden = true; picker.setAttribute("aria-label", "选择参考文件");
      const button = document.createElement("button"); button.type = "button"; button.className = "quiet-button";
      button.textContent = "＋ 添加文件";
      const status = document.createElement("p"); status.className = "chat-caption"; status.setAttribute("role", "status");
      status.textContent = "PDF / Word / Excel / PPT / 文本 · 每份 8 MB；发送问题时读取内容将参与模型提问。";
      const list = document.createElement("div"); list.className = "attachment-list";
      box.append(picker, button, status, list); host.after(box);
      function render() {
        list.replaceChildren();
        for (const item of items) {
          const details = document.createElement("details"), summary = document.createElement("summary");
          summary.textContent = `${item.filename} · 已读取 ${item.characters} 字符${item.warnings?.includes("text_truncated") ? " · 仅纳入前 32,000 字符" : ""}`;
          const preview = document.createElement("pre"); preview.textContent = item.preview;
          const scope = document.createElement("p"); scope.className = "chat-caption";
          scope.textContent = "下方为节选预览；发送时使用本机提取的文本。移除后不再随问题发送，服务器临时副本在会话到期后清理。";
          const remove = document.createElement("button"); remove.type = "button"; remove.className = "quiet-button";
          remove.textContent = "从后续提问移除"; remove.setAttribute("aria-label", `移除文件 ${item.filename}`);
          remove.addEventListener("click", () => {
            if (uploading || !canUpload()) return;
            items = items.filter(value => value.attachment_id !== item.attachment_id); render();
            status.textContent = "已从后续提问移除文件；先前问答不受影响。";
          });
          details.append(summary, scope, preview, remove); list.append(details);
        }
      }
      button.addEventListener("click", () => {
        if (!uploading && canUpload()) picker.click();
        else status.textContent = "请先选择主题，或等待当前回答结束。";
      });
      picker.addEventListener("change", async () => {
        const files = Array.from(picker.files || []); picker.value = "";
        if (uploading || !canUpload()) return;
        if (files.length + items.length > 4) { status.textContent = errors.attachment_capacity; return; }
        uploading = true; button.disabled = true; onBusy(true);
        try {
          const session = await getSession();
          for (const file of files) {
            if (!file.size || file.size > 8 * 1024 * 1024) throw new Error(errors.file_size_limit);
            status.textContent = `正在本机读取 ${file.name}…`;
            const data = new Uint8Array(await file.arrayBuffer());
            let binary = "";
            for (let index = 0; index < data.length; index += 16384) binary += String.fromCharCode(...data.subarray(index, index + 16384));
            const response = await fetch("/api/attachments", { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ channel, session_id: session, filename: file.name, content_base64: btoa(binary) }) });
            const value = await response.json();
            if (!response.ok) throw new Error(errors[value.error] || "文件读取失败，请检查本机服务。");
            items.push(value); render();
          }
          status.textContent = "文件已加入会话，可展开预览。写下问题后发送；连续追问会保留文件内容。";
        } catch (error) { status.textContent = error.message; }
        finally { uploading = false; button.disabled = false; onBusy(false); }
      });
      return {
        ids: () => items.map(item => item.attachment_id), busy: () => uploading,
        reset: (message = "新会话未加入文件。") => { items = []; render(); status.textContent = message; }
      };
    }
  };
})();
