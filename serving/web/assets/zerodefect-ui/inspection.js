
  function formatInspectionBox(box) {
    const values = Array.isArray(box) ? box : [box?.x, box?.y, box?.w, box?.h];
    if (values.length !== 4 || !values.every(Number.isFinite)) return "—";
    const [x, y, w, h] = values.map(value => Number(value.toFixed(2)));
    return `x, y (${x}, ${y}) / ${w} × ${h} px`;
  }

  (() => {
    "use strict";

    const WIDTH = 184;
    const HEIGHT = 597;
    const MAX_BYTES = 5 * 1024 * 1024;
    const THUMB_SCALE = 0.3;
    const IMAGE_TYPES = ["image/png", "image/jpeg"];

    const TYPES = [
      { id: "brown", label: "갈적색결함" },
      { id: "black", label: "흑색결함" },
      { id: "blue", label: "청색결함" },
      { id: "surface", label: "표면결함" }
    ];
    const BUCKETS = [...TYPES, { id: "normal", label: "정상" }];
    const FILTERS = [
      { id: "all", label: "전체" },
      ...BUCKETS,
      { id: "review", label: "재검토" },
      { id: "pending", label: "미검사" },
      { id: "error", label: "실패" }
    ];

    const $ = (id) => document.getElementById(id);

    let items = [];      // 검사 대상: { key, file, state, reason, data, bucket, canvas, badge, btn }
    let skipped = [];    // 제외된 파일: { key, reason }
    let running = false;
    let selected = null;
    const chosenVersions = { detector: "", classifier: "" };
    const deleteSelection = new Set();
    let modelRefreshTimer = null;
    let modelRefreshPending = false;
    const modelListKeys = { detector: "", classifier: "" };
    const modelDropdowns = ["detector", "classifier"];
    let renderToken = 0;
    let adding = false;
    let lastCompletedItem = null;
    let inspectionBatch = null;
    let pauseRequested = false;

    function setStatus(message, isError = false) {
      $("status").textContent = message;
      $("status").hidden = !message;
      $("status").classList.toggle("error", isError);
    }

    async function pool(list, limit, fn) {
      let next = 0;
      const worker = async () => {
        while (next < list.length) {
          await fn(list[next++]);
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(limit, list.length) }, worker)
      );
    }

    const bucketLabel = (id) => id === "review" ? "재검토" : BUCKETS.find((b) => b.id === id)?.label ?? "유형 없음";

    // ---------- 그리기 ----------
    // 결함 박스 색상 설정
    const BOX_COLOR = "#F59E0B";
    const BOX_OUTLINE = "#171717";

    function paint(canvas, bitmap, scale, boxes, defectLabel) {
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));

      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

      if (defectLabel) {
        const labelHeight = 21;
        const labelPadding = 5;
        ctx.font = 'bold 13px "Noto Sans KR", "Malgun Gothic", sans-serif';
        ctx.textBaseline = "middle";

        for (const {x, y, h, score} of boxes) {
          const confidence = Number.isFinite(score) && score >= 0 && score <= 1
            ? ` · ${score.toFixed(2)}` : "";
          const label = defectLabel + confidence;
          const width = Math.min(canvas.width, Math.ceil(ctx.measureText(label).width) + labelPadding * 2);
          const left = Math.min(Math.max(0, x), canvas.width - width);
          const preferredTop = y - labelHeight - 2;
          const top = preferredTop >= 0 ? preferredTop
            : Math.min(y + h + 2, canvas.height - labelHeight);
          ctx.fillStyle = BOX_COLOR;
          ctx.fillRect(left, top, width, labelHeight);
          ctx.fillStyle = BOX_OUTLINE;
          ctx.fillText(label, left + labelPadding, top + labelHeight / 2, width - labelPadding * 2);
        }
      }

      for (const {x, y, w, h} of boxes) {
        ctx.lineWidth = scale === 1 ? 5 : 3;
        ctx.strokeStyle = BOX_OUTLINE;
        ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
        ctx.lineWidth = scale === 1 ? 3 : 1.5;
        ctx.strokeStyle = BOX_COLOR;
        ctx.strokeRect(x * scale, y * scale, w * scale, h * scale);
      }
    }

    async function drawThumb(item) {
      const bitmap = await createImageBitmap(item.file);
      const boxes = item.state === "done" ? item.data.reviewed_boxes ?? item.data.boxes : [];
      paint(item.canvas, bitmap, THUMB_SCALE, boxes);
      bitmap.close();
    }

    function updateBadge(item) {
      const text = {
        ready: "대기",
        running: "검사 중",
        error: "실패",
        "size-error": "크기 오류"
      }[item.state];

      item.btn.dataset.result = item.state === "done" ? (item.data.is_anomaly ? "defect" : "normal") : item.state;
      item.btn.dataset.tone = item.state === "done" ? item.bucket : item.state;
      item.badge.textContent = item.state === "done" ? bucketLabel(item.bucket) : text;
    }

    // ---------- 상세 보기 ----------
    function verdictMessage(item) {
      if (!item) return items.length ? "선택 이미지 없음" : "검사 이미지 없음";
      if (item.state === "ready") return "검사 대기";
      if (item.state === "running") return "검사 중";
      if (item.state === "error") return "검사 실패";
      if (item.state === "size-error") return "이미지 크기 오류";
      if (item.data.reviewed_label) return item.data.reviewed_label === "normal" ? "최종 정상" : "최종 불량";
      return item.data.is_anomaly ? "불량" : "정상";
    }

    function needsReview(item) {
      return item?.state === "done" && item.data.review_required
        && !item.data.reviewed_label && !item.reviewDismissed;
    }

    function inReview(item) {
      return Boolean(item && !item.reviewDismissed && (item.state === "size-error" || needsReview(item)));
    }

    const reviewOverlay = $("review-overlay");
    const reviewType = $("review-defect-type");
    let renderedItem = null;
    let boxDrag = null;
    for (const type of TYPES) reviewType.add(new Option(type.label, type.id));

    function drawReviewBoxes(item) {
      const draft = item?.reviewDraft;
      const saved = item?.data?.reviewed_boxes || [];
      const visible = renderedItem === item && Boolean(draft?.type || saved.length);
      reviewOverlay.toggleAttribute("hidden", !visible);
      reviewOverlay.style.pointerEvents = draft?.type && !reviewType.disabled ? "auto" : "none";
      reviewOverlay.replaceChildren();
      if (!visible) return;
      const boxes = [...(draft?.boxes ?? saved)];
      if (boxDrag?.item === item && boxDrag.preview) {
        if (draft.editing == null) boxes.push(boxDrag.preview);
        else boxes[draft.editing] = boxDrag.preview;
      }
      boxes.forEach((box, index) => {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        for (const [attr, val] of Object.entries({x: box.x, y: box.y, width: box.w, height: box.h})) {
          rect.setAttribute(attr, val);
        }
        if (draft?.editing === index || box === boxDrag?.preview) rect.classList.add("preview");
        reviewOverlay.append(rect);
      });
    }

    function renderReview(item) {
      const show = inReview(item);
      $("final-review").hidden = !show;
      if (!show) {
        drawReviewBoxes(item);
        return;
      }

      const sizeError = item.state === "size-error";
      const busy = running || adding || Boolean(inspectionBatch) || Boolean(item.reviewPending);
      const canJudge = needsReview(item) && Boolean(item.data.id);
      const draft = item.reviewDraft;
      $("review-reason").textContent = sizeError ? item.reason
        : item.data.boxes.length ? "결함 유형을 확인하세요." : "모델이 결함 위치를 찾지 못했습니다.";
      for (const id of ["review-normal", "review-defect"]) {
        $(id).hidden = sizeError;
        $(id).disabled = busy || !canJudge;
      }
      $("review-reinspect").hidden = sizeError;
      $("review-reinspect").disabled = busy || !selectedModelChoice();
      $("review-replace").hidden = !sizeError;
      $("review-replace").disabled = busy;
      $("review-cancel").disabled = busy;
      $("review-status").textContent = item.reviewResult === item.data ? item.reviewMessage || "" : "";
      $("review-annotation").hidden = !draft || sizeError;
      reviewType.value = draft?.type || "";
      reviewType.disabled = busy || !canJudge;
      $("review-save-defect").disabled = busy || !canJudge || !draft?.type || draft.editing != null;
      $("review-box-help").textContent = draft?.editing != null
        ? `박스 ${draft.editing + 1}의 위치를 다시 드래그하세요.`
        : "결함 유형을 선택하세요. 위치를 남기려면 이미지 위에서 박스를 그리세요.";
      const list = $("review-box-list");
      list.replaceChildren();
      for (const [index, box] of (draft?.boxes ?? []).entries()) {
        const row = document.createElement("li");
        row.textContent = formatInspectionBox(box);
        for (const action of ["수정", "삭제"]) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "secondary";
          button.textContent = action;
          button.disabled = busy;
          button.setAttribute("aria-label", `박스 ${index + 1} ${action}`);
          button.addEventListener("click", () => {
            if (action === "수정") {
              draft.editing = draft.editing === index ? null : index;
            } else {
              draft.boxes.splice(index, 1);
              draft.editing = null;
            }
            renderReview(item);
          });
          row.append(button);
        }
        list.append(row);
      }
      drawReviewBoxes(item);
    }

    function reviewPoint(event) {
      const rect = reviewOverlay.getBoundingClientRect();
      return {
        x: Math.round(Math.min(WIDTH, Math.max(0, (event.clientX - rect.left) * WIDTH / rect.width))),
        y: Math.round(Math.min(HEIGHT, Math.max(0, (event.clientY - rect.top) * HEIGHT / rect.height))),
      };
    }

    function boxBetween(start, end) {
      return {x: Math.min(start.x, end.x), y: Math.min(start.y, end.y),
        w: Math.abs(end.x - start.x), h: Math.abs(end.y - start.y)};
    }

    reviewOverlay.addEventListener("pointerdown", (event) => {
      if (boxDrag || event.button !== 0 || !selected?.reviewDraft?.type || reviewType.disabled) return;
      boxDrag = {id: event.pointerId, item: selected, start: reviewPoint(event), preview: null};
      reviewOverlay.setPointerCapture(event.pointerId);
    });
    reviewOverlay.addEventListener("pointermove", (event) => {
      if (boxDrag?.id !== event.pointerId || boxDrag.item !== selected) return;
      boxDrag.preview = boxBetween(boxDrag.start, reviewPoint(event));
      drawReviewBoxes(selected);
    });
    reviewOverlay.addEventListener("pointerup", (event) => {
      if (boxDrag?.id !== event.pointerId || boxDrag.item !== selected) return;
      const box = boxBetween(boxDrag.start, reviewPoint(event));
      boxDrag = null;
      if (box.w && box.h) {
        const draft = selected.reviewDraft;
        if (draft.editing == null) draft.boxes.push(box);
        else draft.boxes[draft.editing] = box;
        draft.editing = null;
      }
      renderReview(selected);
    });
    reviewOverlay.addEventListener("lostpointercapture", () => {
      boxDrag = null;
      drawReviewBoxes(selected);
    });

    async function saveReview(label) {
      const item = selected;
      if (!needsReview(item) || !item.data.id || running || adding || inspectionBatch || item.reviewPending) return;
      const draft = item.reviewDraft;
      if (label === "defect" && (!draft?.type || draft.editing != null)) return;
      const payload = label === "defect" ? {label, defect_type: draft.type, boxes: draft.boxes} : {label};
      const result = item.data;
      item.reviewPending = true;
      item.reviewResult = result;
      item.reviewMessage = "검사 판정 저장 중…";
      renderReview(item);
      try {
        const response = await fetch(`/api/inference/${encodeURIComponent(result.id)}/review`, {
          method: "PUT",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        if (response.status === 401) {
          window.dispatchEvent(new Event("inspection-session-expired"));
          throw new Error("세션이 만료되었습니다.");
        }
        const review = await response.json();
        if (!response.ok) {
          throw new Error(typeof review.detail === "string" ? review.detail : "검사 판정을 저장하지 못했습니다.");
        }
        Object.assign(result, review);
        item.reviewDraft = null;
        item.reviewDismissed = review.review_status === "cancelled";
        item.bucket = bucketOf(result);
        updateBadge(item);
        if (item.reviewDismissed) {
          items = items.filter(entry => entry !== item);
          if (lastCompletedItem === item) lastCompletedItem = null;
          item.row.remove();
          if (selected === item) select(null);
        } else {
          drawThumb(item).catch(() => {});
          if (selected === item) renderViewer();
        }
        renderCounts();
        applyFilter();
        setStatus(item.reviewDismissed ? "검사를 취소했습니다." : "최종 판정을 저장했습니다.");
      } catch (error) {
        item.reviewMessage = error.message;
      } finally {
        item.reviewPending = false;
        syncControls();
        if (item === selected) renderVerdict(item);
      }
    }

    $("review-normal").addEventListener("click", () => saveReview("normal"));
    $("review-defect").addEventListener("click", () => {
      if (!needsReview(selected)) return;
      selected.reviewDraft ??= {type: "", boxes: [], editing: null};
      renderReview(selected);
      reviewType.focus();
    });
    reviewType.addEventListener("change", () => {
      selected.reviewDraft.type = reviewType.value;
      renderReview(selected);
    });
    $("review-save-defect").addEventListener("click", () => saveReview("defect"));
    $("review-reinspect").addEventListener("click", () => {
      if (!needsReview(selected) || $("review-reinspect").disabled) return;
      inspectAll([selected]);
    });
    $("review-cancel").addEventListener("click", () => {
      const item = selected;
      if (!inReview(item) || $("review-cancel").disabled) return;
      if (item.state !== "size-error") {
        saveReview("cancelled");
        return;
      }
      items = items.filter(entry => entry !== item);
      item.row.remove();
      select(null);
      renderCounts();
      applyFilter();
      syncControls();
      setStatus("크기 오류 이미지를 제외했습니다.");
    });

    let replacementTarget = null;
    $("review-replace").addEventListener("click", () => {
      replacementTarget = selected;
      $("review-file").click();
    });
    $("review-file").addEventListener("change", async (event) => {
      const item = replacementTarget;
      replacementTarget = null;
      const [file] = event.target.files;
      event.target.value = "";
      if (!file || !items.includes(item) || item.state !== "size-error" ||
          adding || running || inspectionBatch) return;

      adding = true;
      syncControls();
      try {
        if (!isImageFile(file)) throw new Error("PNG 또는 JPEG 파일을 선택하세요.");
        if (items.some(entry => entry !== item && entry.key === file.name)) {
          throw new Error("같은 이름의 파일이 이미 목록에 있습니다.");
        }
        const corrected = makeItem(withImageType(file), file.name);
        await checkImage(corrected);
        if (corrected.state === "size-error") throw new Error(corrected.reason);
        items[items.indexOf(item)] = corrected;
        item.row.replaceWith(corrected.row);
        if (selected === item) {
          $("type-filter").value = "pending";
          select(null);
        }
        setStatus("파일을 교체했습니다. 검사 시작을 눌러 주세요.");
      } catch (error) {
        item.reviewResult = item.data;
        item.reviewMessage = error.message;
      } finally {
        adding = false;
        renderCounts();
        applyFilter();
        syncControls();
      }
    });

    function verdictRows(item) {
      const result = item?.state === "done" ? item.data : null;
      const prediction = result?.reviewed_label ?? result?.prediction;
      const boxes = result?.reviewed_boxes ?? result?.boxes ?? [];
      const rows = [
        ["결함 종류", !result ? "—" : prediction === "normal" ? "해당 없음" : bucketLabel(item.bucket)],
        ["결함 위치", !result ? "—" : prediction === "normal" ? "해당 없음"
          : boxes.length ? `${boxes.length}곳` : "표시할 위치 없음"],
      ];
      if (!result) return rows;
      if (Object.hasOwn(result, "reviewed_label")) {
        const labels = {normal: "정상", defect: "불량"};
        const review = result.reviewed_label;
        rows.push(["최종 판정", review ? labels[review] ?? review : "미확정"]);
      }
      return rows;
    }

    function renderVerdict(item) {
      const title = $("verdict-message");
      title.textContent = verdictMessage(item);
      title.dataset.tone = item?.state === "done" ? item.bucket : item?.state || "empty";
      $("info").replaceChildren(...verdictRows(item).map(([name, value]) => {
        const row = document.createElement("div");
        const label = document.createElement("dt");
        const detail = document.createElement("dd");
        label.textContent = name;
        detail.textContent = value;
        row.append(label, detail);
        return row;
      }));

      const result = item?.state === "done" ? item.data : null;
      const extra = result
        ? [
          Number.isFinite(result.anomaly_score) && Number.isFinite(result.threshold)
            ? `이상 점수 ${result.anomaly_score.toFixed(4)} / 판정 기준 ${result.threshold.toFixed(4)}` : "",
          !result.boxes.length && result.is_anomaly && Number.isFinite(result.confidence)
            ? `대표 분류 점수 ${result.confidence.toFixed(4)}` : "",
          ...result.boxes.map((box, index) =>
            `영역 ${index + 1} · ${formatInspectionBox(box)} · 분류 점수 ${Number.isFinite(box.score) ? box.score.toFixed(4) : "미기록"}`
          ),
          ...(result.reviewed_boxes ?? []).map((box, index) =>
            `작업자 지정 위치 ${index + 1} · ${formatInspectionBox(box)}`
          ),
          [result.model_name, result.classifier_name].filter(Boolean).join(" / "),
        ]
        : [item?.reason];
      if (item?.reinspectError) extra.push(`재검사 실패 · ${item.reinspectError}`);
      $("verdict-extra").replaceChildren(...extra.filter(Boolean).map((text) => {
        const row = document.createElement("p");
        row.textContent = text;
        return row;
      }));
      renderReview(item);
      $("verdict-details").hidden = !extra.some(Boolean);
      $("json").textContent = result ? JSON.stringify(result, null, 2) : "응답 없음";
    }

    async function renderViewer() {
      const item = selected;
      const token = ++renderToken;
      renderedItem = null;
      const done = item?.state === "done";

      $("viewer-empty").hidden = Boolean(item);
      $("viewer").hidden = !item;
      $("selected-name").textContent = item ? item.key : "선택된 이미지 없음";
      renderVerdict(item);

      if (!item) return;

      const modelBoxes = done && !item.data.reviewed_label ? item.data.boxes : [];
      const visibleBoxes = item.data?.reviewed_boxes ?? modelBoxes;
      $("canvas-boxed").dataset.boxes = String(Boolean(visibleBoxes.length));
      $("canvas-boxed").dataset.tone = done && (item.data.reviewed_label ?? item.data.prediction) === "defect"
        ? "defect" : "";
      const bitmap = await createImageBitmap(item.file);
      if (token !== renderToken) {
        bitmap.close();
        return;
      }

      const scale = Math.min(1, 720 / Math.max(bitmap.width, bitmap.height));
      paint($("canvas-boxed"), bitmap, scale, modelBoxes,
        modelBoxes.length ? bucketLabel(item.bucket) : undefined);
      bitmap.close();
      renderedItem = item;
      drawReviewBoxes(item);
    }

    function select(item) {
      selected?.btn.classList.remove("active");
      selected = item;
      boxDrag = null;
      item?.btn.classList.add("active");
      syncControls();
      return renderViewer();
    }

    // ---------- 리스트와 필터 ----------
    function matches(item, filter) {
      if (item.reviewDismissed) return false;
      if (filter === "review") return inReview(item);
      if (item.state === "size-error") return false;
      if (filter === "all") return true;
      if (filter === "known-defect") return item.state === "done" && TYPES.some(type => type.id === item.bucket);
      if (filter === "incomplete") return ["ready", "running"].includes(item.state);
      if (filter === "pending") return item.state === "ready";
      if (filter === "error") return item.state === "error";
      return item.state === "done" && item.bucket === filter;
    }

    function applyFilter() {
      const filter = $("type-filter").value;
      $("category-select").value = filter;
      let shown = 0;

      for (const item of items) {
        const ok = matches(item, filter);
        item.row.hidden = !ok;
        if (ok) shown += 1;
      }

      document.querySelectorAll("[data-category]").forEach(b => b.setAttribute("aria-pressed",String(b.dataset.category===filter)));
      $("list-empty").hidden = shown > 0;
      $("list-empty").textContent = items.length
        ? "해당 종류의 이미지가 없습니다."
        : "선택한 이미지가 없습니다.";
    }

    function buildFilter() {
      const categorySelect = $("category-select");
      categorySelect.replaceChildren(...FILTERS.map(({id, label}) => new Option(label, id)));
      categorySelect.addEventListener("change", () => {
        $("type-filter").value = categorySelect.value;
        applyFilter();
      });
      for (const f of FILTERS.filter(({id}) => id !== "all")) {
        const button=document.createElement("button");button.type="button";button.dataset.category=f.id;
        button.addEventListener("click", () => {
          const filter = $("type-filter");
          filter.value = filter.value === f.id ? "all" : f.id;
          applyFilter();
        });$("category-tabs").append(button);

      }
    }

    function renderCounts() {
      const listCount = $("current-image-count");
      if (listCount) listCount.textContent = `현재 목록 ${items.length}장`;
      const total = items.filter(item => item.state !== "size-error").length;
      const normal = items.filter(item => item.state === "done" && item.bucket === "normal").length;
      const review = items.filter(item => item.state === "done" && item.bucket === "review").length;
      const defect = items.filter(item => item.state === "done" && TYPES.some(type => type.id === item.bucket)).length;
      const failed = items.filter(item => item.state === "error").length;
      const processed = normal + review + defect + failed;
      const share = count => total ? (100 * count / total).toFixed(1).replace(/\.0$/, "") : "0";
      const summary = document.createElement("strong");
      summary.textContent = `전체 처리 ${share(processed)}% (${processed}/${total})`;
      const gauge = document.createElement("div");
      gauge.className = "inspection-gauge";
      gauge.setAttribute("aria-label", "검사 상태별 비율");
      const legend = document.createElement("div");
      legend.className = "inspection-legend";
      for (const [label, count, color, filter] of [
        ["정상", normal, "#237a50", "normal"],
        ["불량 의심 · 유형 확인", defect, "#be3e36", "known-defect"],
        ["재검토", review, "#ad6800", "review"],
        ["검사 실패", failed, "#7041a1", "error"],
        ["처리 전·진행 중", total - processed, "#e5e9ef", "incomplete"],
      ]) {
        const segment = document.createElement("span");
        const entry = document.createElement("span");
        const caption = `${label} ${count}장 · ${share(count)}%`;
        segment.hidden = count === 0;
        segment.style.flexBasis = total ? `${100 * count / total}%` : "0%";
        segment.style.backgroundColor = color;
        gauge.append(segment);
        const swatch = document.createElement("i");
        swatch.style.backgroundColor = color;
        entry.title = caption;
        entry.setAttribute("aria-label", caption);
        const shortLabels = {normal: "정상", "known-defect": "불량", review: "재검토", error: "실패", incomplete: "대기"};
        const metric = document.createElement("small");
        metric.textContent = ` ${share(count)}% (${count})`;
        entry.append(swatch, shortLabels[filter], metric);
        legend.append(entry);
      }
      $("counts").replaceChildren(summary, gauge, legend);

      for (const [index, {id, label}] of FILTERS.entries()) {
        const count = items.filter(item => matches(item, id)).length;
        const caption = `${label} (${count})`;
        $("category-select").options[index].textContent = caption;
        const button = $("category-tabs").querySelector(`[data-category="${id}"]`);
        if (button) button.textContent = caption;
      }

    }

    function renderSummary() {
      $("skipped").hidden = skipped.length === 0;
      $("skipped-summary").textContent = `제외된 파일 (${skipped.length})`;
      $("skipped-list").replaceChildren(
        ...skipped.slice(0, 30).map((s) => {
          const li = document.createElement("li");
          li.textContent = `${s.key} — ${s.reason}`;
          return li;
        })
      );
      if (skipped.length > 30) {
        const li = document.createElement("li");
        li.textContent = `외 ${skipped.length - 30}개`;
        $("skipped-list").append(li);
      }
    }

    function syncControls() {
      const choice = selectedModelChoice();
      const pending = items.some((item) => item.state !== "error" && needsInspection(item, choice));
      const reinspect = !inspectionBatch && !pending && items.some(canReinspect);
      const inspectButton = $("inspect-button");

      inspectButton.disabled = running || adding || !choice || (!inspectionBatch && !pending && !reinspect);
      inspectButton.dataset.action = reinspect ? "reinspect" : "inspect";
      const reinspectButton = $("reinspect-button");
      reinspectButton.hidden = Boolean(inspectionBatch) || !pending || !items.some(canReinspect);
      reinspectButton.disabled = running || adding || !choice;
      const action = $("inspection-action");
      action.hidden = !inspectionBatch;
      action.disabled = adding || pauseRequested;
      action.textContent = running ? (pauseRequested ? "중단 처리 중" : "검사 중단") : "검사 취소";
      $("download-image").disabled = !items.some(item => ["done", "error"].includes(item.state));
      inspectButton.textContent = adding ? "이미지 확인 중" : running ? "검사 실행 중" : inspectionBatch ? "검사 재개" : !items.length ? "검사 이미지 없음" : !choice ? "모델 버전 선택" : reinspect ? "재검사" : "검사 시작";

      $("pick-files").disabled = running || adding;
      $("clear-button").hidden = !action.hidden;
      $("clear-button").disabled = adding || !items.length;
      renderReview(selected);
      for (const role of modelDropdowns) {
        const dropdown = $(`${role}-dropdown`);
        const locked = running || adding;
        dropdown.inert = locked;
        dropdown.setAttribute("aria-disabled", String(locked));
        if (locked) dropdown.open = false;
      }
    }

    // ---------- 파일 추가 ----------
    function makeItem(file, key) {
      const item = { key, file, state: "ready", reason: "", data: null, bucket: null };

      item.canvas = document.createElement("canvas");
      item.badge = document.createElement("span");
      item.badge.className = "badge";
      item.badge.textContent = "대기";

      item.btn = document.createElement("button");
      item.btn.type = "button";
      item.btn.className = "thumb";
      item.btn.title = key;
      item.btn.setAttribute("aria-label", `${key} 보기`);
      const filename = document.createElement("span");
      filename.className = "filename";
      filename.textContent = key;
      item.btn.append(item.canvas, filename, item.badge);
      item.btn.addEventListener("click", () => select(item));
      item.row = document.createElement("div");
      item.row.className = "image-row";
      item.row.append(item.btn);
      return item;
    }

    async function checkImage(item) {
      if (item.file.size > MAX_BYTES) throw new Error("5MB 초과");

      let bitmap;
      try {
        bitmap = await createImageBitmap(item.file);
      } catch {
        throw new Error("읽을 수 없는 이미지");
      }

      try {
        if (bitmap.width !== WIDTH || bitmap.height !== HEIGHT) {
          item.state = "size-error";
          item.reason = `현재 ${bitmap.width}×${bitmap.height}px · 필요 ${WIDTH}×${HEIGHT}px`;
          updateBadge(item);
          paint(item.canvas, bitmap, 180 / Math.max(bitmap.width, bitmap.height), []);
          return;
        }
        paint(item.canvas, bitmap, THUMB_SCALE, []);
      } finally {
        bitmap.close();
      }
    }

    function isImageFile(file) {
      return IMAGE_TYPES.includes(file.type) || (!file.type && /\.(png|jpe?g)$/i.test(file.name));
    }

    function withImageType(file) {
      if (file.type) return file;
      const type = /\.png$/i.test(file.name) ? "image/png" : "image/jpeg";
      return new File([file], file.name, {type, lastModified: file.lastModified});
    }

    async function addFiles(files) {
      if (running || adding) {
        setStatus("현재 검사 재개 후 이미지 추가 가능", true);
        return;
      }

      adding = true; syncControls();
      const known = new Set(items.map(item => item.key));
      const excluded = new Set(skipped.map(item => item.key));
      const groups = new Map();
      const fresh = [];

      for (const file of files) {
        const key = file.relativePath || file.webkitRelativePath || file.name;
        const folder = key.includes("/") ? key.split("/", 1)[0] : "";
        if (!groups.has(folder)) {
          groups.set(folder, {added: 0, duplicate: 0, sizeError: 0, rejected: 0});
        }
        const counts = groups.get(folder);
        if (!isImageFile(file) || excluded.has(key)) {
          counts.rejected += 1;
          continue;
        }
        if (known.has(key)) {
          counts.duplicate += 1;
          continue;
        }

        known.add(key);
        fresh.push({item: makeItem(withImageType(file), key), counts});
      }

      setStatus("");

      await pool(fresh, 8, async ({item, counts}) => {
        try {
          await checkImage(item);
          items.push(item);
          if (item.state === "size-error") counts.sizeError += 1;
          else {
            counts.added += 1;
            inspectionBatch?.targets.push(item);
          }
        } catch (error) {
          counts.rejected += 1;
          skipped.push({ key: item.key, reason: error.message });
        }
      });

      const addedCount = $("added-image-count");
      const lines = Array.from(groups, ([folder, counts]) => {
        const line = document.createElement("p");
        const parts = [folder || "개별 파일", `신규 ${counts.added}장`];
        if (counts.duplicate) parts.push(`이미 등록 ${counts.duplicate}장`);
        if (counts.sizeError) parts.push(`크기 오류 ${counts.sizeError}장`);
        if (counts.rejected) parts.push(`제외 ${counts.rejected}개`);
        line.textContent = parts.join(" · ");
        return line;
      });
      const listCount = document.createElement("p");
      listCount.id = "current-image-count";
      addedCount.replaceChildren(...lines, listCount);
      addedCount.hidden = false;
      const sizeErrors = fresh.some(({item}) => item.state === "size-error");
      items.sort((a, b) => a.key.localeCompare(b.key));
      $("list").replaceChildren(...items.map((i) => i.row));

      adding = false;
      if (sizeErrors && !items.some(item => item.state === "ready")) $("type-filter").value = "review";
      renderSummary();
      renderCounts();
      applyFilter();
      syncControls();

      if (items.length) {
        setStatus("");
      } else {
        setStatus("검사할 수 있는 이미지가 없습니다.", true);
      }
    }

    function updateDeleteCount() {
      const count = deleteSelection.size;
      $("delete-count").textContent = `선택 ${count}장`;
      $("confirm-delete").disabled = count === 0;
      $("confirm-delete").textContent = count ? `선택한 ${count}장 삭제` : "이미지 삭제";
    }

    function configureImageSelection(prefix, container, candidates, filters) {
      const filter = $(`${prefix}-filter`);
      const options = $(container);
      const selectAll = $(`select-all-${prefix}`);
      filter.replaceChildren(...filters.filter(({id}) => id === "all" || candidates.some(item => matches(item, id))).map(({id, label}) => new Option(label, id)));
      function visibleChecks() {
        return Array.from(options.querySelectorAll("label:not([hidden]) input"));
      }
      function updateSelectAll() {
        const checks = visibleChecks();
        selectAll.disabled = checks.length === 0;
        const allChecked = checks.length > 0 && checks.every(input => input.checked);
        selectAll.textContent = allChecked ? "전체 해제" : "전체 선택";
      }
      filter.onchange = () => {
        Array.from(options.children).forEach((label, index) => {
          label.hidden = !matches(candidates[index], filter.value);
        });
        updateSelectAll();
      };
      selectAll.onclick = () => {
        const checks = visibleChecks();
        const checked = checks.some(input => !input.checked);
        for (const input of checks) {
          input.checked = checked;
          input.dispatchEvent(new Event("change", {bubbles: true}));
        }
      };
      options.onchange = updateSelectAll;
      filter.onchange();
    }

    function canReinspect(item) {
      return !item.reviewDismissed && !item.reviewPending && (item.state === "done" || item.state === "error");
    }

    function imageSelectionOption(item, index) {
      const label = document.createElement("label");
      label.className = "download-option";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.value = index;
      const thumb = document.createElement("canvas");
      thumb.width = item.canvas.width;
      thumb.height = item.canvas.height;
      if (thumb.width && thumb.height) thumb.getContext("2d").drawImage(item.canvas, 0, 0);
      const name = document.createElement("span");
      name.textContent = item.key;
      label.append(check, thumb, name);
      return label;
    }

    function openDeleteDialog() {
      if (running || adding || inspectionBatch || !items.length) return;
      deleteSelection.clear();
      $("delete-options").replaceChildren(...items.map((item, index) => {
        const label = imageSelectionOption(item, index);
        const check = label.querySelector("input");
        check.addEventListener("change", () => {
          if (check.checked) deleteSelection.add(item);
          else deleteSelection.delete(item);
          updateDeleteCount();
        });
        return label;
      }));
      updateDeleteCount();
      configureImageSelection("delete", "delete-options", items, FILTERS);
      $("delete-dialog").showModal();
    }

    function deleteSelectedImages() {
      if (running || adding || !deleteSelection.size) return;

      if ([...deleteSelection].some(item => item.reviewPending)) return;
      const count = deleteSelection.size;
      const removedSelected = deleteSelection.has(selected);
      items = items.filter((item) => !deleteSelection.has(item));
      for (const item of deleteSelection) {
        if (lastCompletedItem === item) lastCompletedItem = null;
        item.row.remove();
      }
      if (removedSelected) select(null);

      $("delete-dialog").close();
      renderSummary();
      renderCounts();
      applyFilter();
      syncControls();
      setStatus(`${count}장 삭제됨`);
    }

    // ---------- 검사 ----------
    function normalize(data) {
      if (
        typeof data.is_anomaly !== "boolean" ||
        !Number.isFinite(data.anomaly_score) ||
        !Number.isFinite(data.threshold)
      ) {
        throw new Error("API 응답 형식이 올바르지 않습니다.");
      }

      const boxes = (Array.isArray(data.boxes) ? data.boxes : []).filter(
        (b) => b && [b.x, b.y, b.w, b.h].every(Number.isFinite) && b.w > 0 && b.h > 0
      );

      return { ...data, boxes };
    }

    function bucketOf(data) {
      if ((data.reviewed_label ?? data.prediction) === "normal") return "normal";
      if (data.review_required && !data.reviewed_label) return "review";
      const type = data.reviewed_defect_type ?? data.defect_type;
      return TYPES.some(option => option.id === type) ? type : "review";
    }

    let healthRequest = 0;

    function setInspectionServiceState(message, tone = "") {
      const label = $("inspection-service-label");
      label.textContent = message;
      label.dataset.tone = tone;
      $("inspection-service-spinner").hidden = tone !== "loading";
      $("inspection-service-recheck").hidden = tone !== "connection";
    }

    async function checkInspectionService() {
      if (running) return;
      const request = ++healthRequest;
      setInspectionServiceState("검사 서버 확인 중");

      try {
        const response = await fetch("/api/health", {
          cache: "no-store",
          signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) throw new Error("Health request failed");
        const health = await response.json();
        if (typeof health.model_loaded !== "boolean" ||
            typeof health.classifier_loaded !== "boolean") {
          throw new Error("Invalid health response");
        }
        if (request !== healthRequest) return;
        setInspectionServiceState(selectedModelChoice() || inspectionBatch?.choice
          ? "검사 서버 연결됨 · 시작 시 모델 확인"
          : "검사 서버 연결됨 · 모델 버전 선택");
      } catch {
        if (request === healthRequest) {
          setInspectionServiceState("검사 서버 상태 확인 실패", "connection");
        }
      }
    }

    function inspectionRequestError(response, message) {
      const error = new Error(message);
      error.httpStatus = response.status;
      return error;
    }

    async function predict(file, choice) {
      const form = new FormData();
      form.append("file", file);
      form.append("model_choice", choice);

      const response = await fetch("/api/predict", { method: "POST", body: form });
      if (response.status === 401) {
        throw inspectionRequestError(response, "다시 로그인한 뒤 검사 재개를 누르세요.");
      }
      const body = await response.text();

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${body.slice(0, 200)}`);
      }
      return normalize(JSON.parse(body));
    }

    async function prepareRegistryModel(choice) {
      if (!choice.startsWith("registry-")) return true;

      setInspectionServiceState("선택 모델 확인 중…", "loading");
      const deadline = Date.now() + 300000;
      while (!pauseRequested && Date.now() < deadline) {
        const response = await fetch(`/api/models/${encodeURIComponent(choice)}/prepare`, {
          method: "POST",
          signal: AbortSignal.timeout(15000),
        });
        if (!response.ok) {
          const message = response.status === 409
            ? "선택한 모델 버전을 다시 확인하세요."
            : `모델 준비 요청 실패 (HTTP ${response.status})`;
          throw inspectionRequestError(response, message);
        }

        const { status } = await response.json();
        if (status === "ready") {
          if (!pauseRequested) setInspectionServiceState("선택 모델 준비 완료", "ready");
          return !pauseRequested;
        }
        if (status === "failed") throw new Error("모델 로드 실패");
        if (!["loading", "busy"].includes(status)) throw new Error("모델 준비 상태 확인 실패");
        setInspectionServiceState(status === "busy" ? "다른 모델 준비가 끝나기를 기다리는 중…" : "선택 모델을 불러오는 중…", "loading");

        await new Promise((resolve) => setTimeout(resolve, 3000));
      }
      if (pauseRequested) return false;
      throw new Error("모델 준비 시간 초과");
    }

    async function runInspectionItem(item, choice) {
      const previousResult = item.state === "done";
      if (!previousResult) item.state = "running";
      item.reason = "";
      item.reinspectError = "";
      updateBadge(item);
      if (item === selected) renderVerdict(item);
      renderCounts();
      syncControls();

      try {
        item.data = await predict(item.file, choice);
        item.reviewDismissed = false;
        item.reviewDraft = null;
        item.lastChoice = choice;
        item.bucket = bucketOf(item.data);
        item.state = "done";
      } catch (error) {
        if (previousResult) item.reinspectError = error.message;
        else {
          item.state = "error";
          item.reason = error.message;
        }
        if (error.httpStatus === 401) {
          updateBadge(item);
          if (item === selected) renderVerdict(item);
          throw error;
        }
      }

      updateBadge(item);
      const completed = item.state === "done" && !item.reinspectError;
      if (completed) {
        lastCompletedItem = item;
        try { await drawThumb(item); } catch { /* Keep the inference result. */ }
      }

      renderCounts();
      applyFilter();
      const following = completed && followItem(item);
      if (!following && item === selected) renderViewer();
      syncControls();
    }

    function showInspectionProgress(batch) {
      setInspectionServiceState(`검사 진행 중 · ${batch.processed} / ${batch.targets.length}장`);
    }

    async function inspectAll(selectedTargets) {
      if (running || adding) return;
      const resuming = Boolean(inspectionBatch);
      const choice = selectedModelChoice();
      if (!choice) return;

      const targets = inspectionBatch?.targets || selectedTargets || items.filter((item) => needsInspection(item, choice));
      if (!targets.length) return;

      const batch = inspectionBatch || {choice, targets, next: 0, processed: 0};
      inspectionBatch = batch;
      pauseRequested = false;
      running = true;
      ++healthRequest;
      syncControls();
      renderCounts();

      try {
        const prepared = await prepareRegistryModel(choice);
        if (prepared && !pauseRequested) {
          batch.choice = choice;
          setStatus("");
          showInspectionProgress(batch);
          while (!pauseRequested && batch.next < batch.targets.length) {
            const item = batch.targets[batch.next];
            await runInspectionItem(item, batch.choice);
            batch.next += 1;
            batch.processed += 1;
            showInspectionProgress(batch);

          }
        }
        if (batch.next >= batch.targets.length) {
          inspectionBatch = null;
          setInspectionServiceState(`검사 처리 완료 · ${batch.processed} / ${batch.targets.length}장`);
        }
        setStatus("");
      } catch (error) {
        if (error.httpStatus === 401) {
          const message = "다시 로그인한 뒤 검사 재개를 누르세요.";
          setInspectionServiceState(message, "error");
          setStatus(message, true);
          window.dispatchEvent(new Event("inspection-session-expired"));
        } else if (!pauseRequested) {
          if (!resuming) inspectionBatch = null;
          setInspectionServiceState(error.message, "error");
          setStatus(error.message, true);
        }
      } finally {
        if (pauseRequested && inspectionBatch) setInspectionServiceState("검사 일시 중단");
        pauseRequested = false;
        running = false;
        syncControls();
        renderCounts();
      }
    }

    // ---------- 모델 정보 ----------
    function selectedModelChoice() {
      const { detector, classifier } = chosenVersions;
      return detector && classifier ? `registry-d${detector}-c${classifier}` : "";
    }

    function needsInspection(item, choice) {
      return !item.reviewDismissed && !item.reviewPending && (
        item.state === "ready" || item.state === "error" ||
        (item.state === "done" && choice && item.lastChoice !== choice)
      );
    }

    function renderVersionOptions(role, versions) {
      const entries = versions.map(({ version, status }) => ({version: String(version), status}));
      if (JSON.stringify(entries) === modelListKeys[role]) return;
      modelListKeys[role] = JSON.stringify(entries);

      const selected = entries.find((entry) => entry.version === chosenVersions[role]);
      if (!selected || selected.status !== "READY") chosenVersions[role] = "";
      $(`${role}-label`).textContent = chosenVersions[role] ? `v${chosenVersions[role]}` : "버전 선택";

      const options = $(`${role}-options`);
      if (!entries.length) {
        options.textContent = "등록된 버전이 없습니다.";
        return;
      }

      options.replaceChildren(...entries.map(({version, status}) => {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = `${role}-version`;
        input.checked = chosenVersions[role] === version;
        input.disabled = status !== "READY";
        input.addEventListener("change", () => {
          chosenVersions[role] = version;
          $(`${role}-label`).textContent = `v${version}`;
          $(`${role}-dropdown`).open = false;
          $(`${role}-dropdown`).querySelector("summary").focus();
          syncControls();
          if (inspectionBatch) {
            ++healthRequest;
            setInspectionServiceState("검사 일시 중단 · 재개 시 선택 모델 확인");
          } else {
            checkInspectionService();
          }
        });
        label.append(input, status === "READY" ? `v${version}` : `v${version} · ${status}`);
        return label;
      }));
    }

    async function loadModelInfo() {
      if (modelRefreshPending) return;
      modelRefreshPending = true;
      const message = $("model-info");

      try {
        const response = await fetch("/api/models/versions", {
          cache: "no-store",
          signal: AbortSignal.timeout(8000),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data.detectors) || !Array.isArray(data.classifiers)) {
          throw new Error("Invalid model version list");
        }
        renderVersionOptions("detector", data.detectors);
        renderVersionOptions("classifier", data.classifiers);
        message.hidden = true;
        syncControls();
      } catch {
        message.textContent = "모델 버전 목록 조회 실패 · 목록 다시 열기";
        message.hidden = false;
      } finally {
        modelRefreshPending = false;
      }
    }

    for (const role of modelDropdowns) {
      const dropdown = $(`${role}-dropdown`);
      dropdown.addEventListener("toggle", () => {
        if (dropdown.open) {
          loadModelInfo();
          if (modelRefreshTimer === null) modelRefreshTimer = setInterval(loadModelInfo, 10000);
        } else if (!modelDropdowns.some((name) => $(`${name}-dropdown`).open)) {
          clearInterval(modelRefreshTimer);
          modelRefreshTimer = null;
        }
      });
    }

    $("inspection-action").addEventListener("click", () => {
      if (adding || !inspectionBatch || pauseRequested) return;
      if (running) {
        pauseRequested = true;
        setStatus("");
        syncControls();
        return;
      }
      inspectionBatch = null;
      setStatus("");
      renderCounts();
      syncControls();
    });
    $("swap-inspection-panels").addEventListener("click", event => {
      const left = $("inspection-left");
      const center = $("inspection-center");
      const leftPanel = left.firstElementChild;
      const centerPanel = center.firstElementChild;
      left.append(centerPanel);
      center.append(leftPanel);
      const listInCenter = center.contains($("list"));
      event.currentTarget.setAttribute("aria-pressed", String(listInCenter));
      $("swap-inspection-label").textContent = listInCenter ? "목록을 왼쪽으로" : "목록을 가운데로";
    });
    let exportImages = [];
    $("download-image").addEventListener("click", () => {
      exportImages = items.filter(item => ["done", "error"].includes(item.state))
        .map(item => ({
          item,
          result: item.state === "error"
            ? {file: item.key, status: "error", reason: item.reason}
            : {file: item.key, ...item.data},
        }));
      if (!exportImages.length) return;
      $("download-options").replaceChildren(...exportImages.map(({item}, index) => {
        const label = document.createElement("label");
        label.className = "download-option";
        const check = document.createElement("input");
        check.type = "checkbox";
        check.value = String(index);
        check.checked = item === selected;
        const thumb = document.createElement("canvas");
        thumb.width = item.canvas.width;
        thumb.height = item.canvas.height;
        thumb.setAttribute("aria-hidden", "true");
        if (thumb.width && thumb.height) thumb.getContext("2d").drawImage(item.canvas, 0, 0);
        label.append(check, thumb, item.key);
        return label;
      }));
      $("confirm-export").disabled = !$("download-options").querySelector("input:checked");
      configureImageSelection("export", "download-options", exportImages.map(({item}) => item), FILTERS.filter(({id}) => id !== "pending"));
      $("export-dialog").showModal();
    });
    $("download-options").addEventListener("change", () => {
      $("confirm-export").disabled = !$("download-options").querySelector("input:checked");
    });
    $("confirm-export").addEventListener("click", () => {
      const choices = Array.from($("download-options").querySelectorAll("input:checked"),
        input => exportImages[Number(input.value)]);
      if (!choices.length) return;
      const [first] = choices;
      const single = choices.length === 1;
      const content = JSON.stringify(single ? first.result : choices.map(choice => choice.result), null, 2);
      const url = URL.createObjectURL(new Blob([content], {type: "application/json"}));
      const link = document.createElement("a");
      link.href = url;
      link.download = single ? first.item.file.name.replace(/\.[^.]+$/, "") + ".json" : "inspection-results.json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      $("export-dialog").close();
    });

    function followItem(item) {
      if (!$("follow-progress").checked || item.row.hidden) return false;
      const list = $("list"), box = list.getBoundingClientRect(), row = item.row.getBoundingClientRect();
      if (row.bottom > box.bottom) list.scrollTop += row.bottom - box.bottom;
      else if (row.top < box.top) list.scrollTop += row.top - box.top;
      select(item);
      return true;
    }
    $("follow-progress").addEventListener("change", () => {
      if (!$("follow-progress").checked) return;
      if (lastCompletedItem?.state === "done") followItem(lastCompletedItem);
    });
    const drop = $("drop-zone");
    drop.addEventListener("dragover",e=>{e.preventDefault();drop.classList.add("drag-over");});
    drop.addEventListener("dragleave", event => {
      if (!drop.contains(event.relatedTarget)) drop.classList.remove("drag-over");
    });
    async function readEntry(entry, prefix = "") {
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        Object.defineProperty(file, "relativePath", {value: prefix + file.name});
        return [file];
      }
      if (!entry.isDirectory) return [];
      const reader = entry.createReader();
      const files = [];
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) return files;
        for (const child of batch) files.push(...await readEntry(child, prefix + entry.name + "/"));
      }
    }

    drop.addEventListener("drop", async event => {
      event.preventDefault();
      drop.classList.remove("drag-over");
      if (running || adding) return;
      $("added-image-count").hidden = true;
      const sources = Array.from(event.dataTransfer.items)
        .filter(item => item.kind === "file")
        .map(item => ({entry: item.webkitGetAsEntry?.(), file: item.getAsFile()}));
      const fallback = Array.from(event.dataTransfer.files);
      adding = true;
      syncControls();
      try {
        const files = [];
        for (const {entry, file} of sources) {
          if (entry) files.push(...await readEntry(entry));
          else if (file) files.push(file);
        }
        adding = false;
        await addFiles(sources.length ? files : fallback);
      } catch (error) {
        setStatus(`파일 읽기 실패: ${error.message}`, true);
      } finally {
        adding = false;
        syncControls();
      }
    });
    // ---------- 시작 ----------
    $("pick-files").addEventListener("click", () => $("files").click());

    async function handlePickedImages(event) {
      const list = Array.from(event.target.files);
      event.target.value = "";
      await addFiles(list);
    }
    $("files").addEventListener("change", handlePickedImages);

    $("inspect-button").addEventListener("click", () => {
      if ($("inspect-button").dataset.action === "reinspect") openReinspectDialog();
      else inspectAll();
    });

    $("reinspect-button").addEventListener("click", openReinspectDialog);

    function openReinspectDialog() {
      if (running || adding || inspectionBatch || !selectedModelChoice()) return;
      const candidates = items.filter(canReinspect);
      if (!candidates.length) return;

      const options = $("reinspect-options");
      const confirm = $("confirm-reinspect");
      options.replaceChildren(...candidates.map(imageSelectionOption));
      function updateCount() {
        const count = options.querySelectorAll("input:checked").length;
        $("reinspect-count").textContent = `선택 ${count}장`;
        confirm.disabled = count === 0;
      }
      configureImageSelection("reinspect", "reinspect-options", candidates, FILTERS);
      const updateSelectAll = options.onchange;
      options.onchange = () => {
        updateSelectAll();
        updateCount();
      };
      confirm.onclick = () => {
        const targets = Array.from(options.querySelectorAll("input:checked"), input => candidates[Number(input.value)]);
        if (!targets.length) return;
        $("reinspect-dialog").close();
        inspectAll(targets);
      };
      updateCount();
      $("reinspect-dialog").showModal();
    }
    $("cancel-reinspect").addEventListener("click", () => $("reinspect-dialog").close());
    $("inspection-service-recheck").addEventListener("click", checkInspectionService);
    document.querySelector('[data-tab="inspect"]').addEventListener("click", checkInspectionService);
    window.addEventListener("session-ready", checkInspectionService);
    $("clear-button").addEventListener("click", openDeleteDialog);
    $("confirm-delete").addEventListener("click", deleteSelectedImages);
    $("cancel-delete").addEventListener("click", () => $("delete-dialog").close());
    $("delete-dialog").addEventListener("close", () => {
      deleteSelection.clear();
      updateDeleteCount();
    });

    buildFilter();
    renderSummary();
    renderCounts();
    applyFilter();
    syncControls();
    renderViewer();
    window.addEventListener("session-ready", loadModelInfo, { once: true });
  })();
