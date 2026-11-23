
    (() => {
      const form = document.getElementById("login-form");
      const status = document.getElementById("login-status");
      const screen = document.getElementById("inspection-screen");
      const inviteForm = document.getElementById("invite-form");
      const registerForm = document.getElementById("register-form");
      const registerStatus = document.getElementById("register-status");
      let inviteToken = new URLSearchParams(window.location.hash.slice(1)).get("invite");
      const submit = form.querySelector('button[type="submit"]');

      let currentRole = null;
      const tabs = [...document.querySelectorAll("[data-tab]")];
      const monitoringViews = document.getElementById("monitoring-views");
      const monitorPanel = document.getElementById("panel-monitor");
      const monitorViews = [...monitoringViews.querySelectorAll("[data-monitor-view]")].map(button => {
        const panel = document.getElementById(button.getAttribute("aria-controls"));
        return { button, panel, frames: panel.querySelectorAll("iframe[data-src]") };
      });
      let selectedMonitorView = "worker";

      function canOpenTab(name) {
        if (name === "inspect" || name === "history") return true;
        if (name === "monitor") return currentRole === "worker" || currentRole === "admin";
        return currentRole === "admin";
      }

      const training = (() => {
        const element = (id) => document.getElementById(`training-${id}`);
        const form = element("form");
        const model = element("model");
        const fold = element("fold");
        const register = element("register");
        const start = element("start");
        const stop = element("stop");
        const feedback = element("feedback");
        const current = element("current");
        const history = element("history");
        const empty = element("empty");
        const log = element("log");
        const logTitle = element("log-title");
        const detail = element("detail");
        const retry = element("retry");
        const progressBar = element("progress-bar");
        const progressPercent = element("progress-percent");
        const progressPhase = element("progress-phase");
        const logFacts = new Map();
        const models = { anomaly: "이상치 탐지", classifier: "결함 분류" };
        const states = {
          queued: "대기 중", running: "실행 중", stopping: "중단 중", succeeded: "완료",
          failed: "실패", cancelled: "중단됨", busy: "다른 학습 실행 중", unknown: "상태 확인 필요",
        };
        const dates = new Intl.DateTimeFormat("ko-KR", {
          dateStyle: "short", timeStyle: "medium", timeZone: "Asia/Seoul",
        });
        let visible = false;
        let timer;
        let request;
        let session = 0;
        let ready = false;
        let busy = false;
        let activeJob = null;
        let picked = null;
        let selectedJob = null;
        let displayedLogJob = null;
        let previousLogJob = null;

        function duration(seconds) {
          if (!Number.isInteger(seconds) || seconds < 0) return "—";
          const hours = Math.floor(seconds / 3600);
          const minutes = Math.floor(seconds / 60) % 60;
          return `${hours ? hours + "시간 " : ""}${minutes}분 ${seconds % 60}초`;
        }

        function readLogFacts(text) {
          const pattern = /^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)\] (시작:|종료:)/gm;
          let begin, end;
          for (const match of text.matchAll(pattern)) {
            const timestamp = Date.parse(`${match[1]}T${match[2]}Z`);
            if (match[3] === "시작:") begin = timestamp;
            else end = timestamp;
          }
          const fold = text.match(/^\[[^\n]+\] 시작:[^\n]*(?:=|\s)--fold(?:=|\s+)(\d+)\b/m)?.[1];
          const runId = text.match(/View run[^\n]*\/(?:runs)\/([0-9a-f]{32})(?![0-9a-f])/i)?.[1];
          const facts = {};
          if (Number.isFinite(begin)) {
            facts.begin = begin;
            facts.phase = "학습 준비 중";
            facts.percent = null;
          }
          if (Number.isFinite(end)) facts.end = end;
          if (fold !== undefined) facts.fold = Number(fold);
          if (runId) facts.runId = runId;
          for (const sourceLine of text.split("\n")) {
            const line = sourceLine.trim();
            const coreset = line.match(/^coreset\s+(\d+)\/(\d+)$/);
            const epoch = line.match(/^ep\s*(\d+)\s+lr=/);
            const fineTune = line.match(/^epoch\s+(\d+).*val_AP=/);
            if (coreset && Number(coreset[2]) > 0 && Number(coreset[1]) <= Number(coreset[2])) {
              facts.phase = `코어셋 단계 · ${coreset[1]} / ${coreset[2]}`;
              facts.percent = Math.floor(100 * Number(coreset[1]) / Number(coreset[2]));
              continue;
            }
            let phase;
            if (epoch) phase = `결함 분류 학습 · Epoch ${epoch[1]} 완료`;
            else if (fineTune) phase = `이상치 탐지 학습 · Epoch ${fineTune[1]} 평가 완료`;
            else if (line.startsWith("coreset 후보 패치:")) phase = "코어셋 구성 중";
            else if (line.startsWith("파인튜닝 stage")) phase = "파인튜닝 중";
            else if (/^(memory bank 크기:|임계값|\[(val|test)\]|\[fold\d+\]\s+best|MLflow 기록 완료:)/.test(line)) {
              phase = "평가 및 결과 저장 중";
            }
            if (phase) {
              facts.phase = phase;
              facts.percent = null;
            }
          }
          return facts;
        }

        function jobDuration(job) {
          if (Number.isInteger(job.duration_seconds) && job.duration_seconds >= 0) {
            return job.duration_seconds;
          }
          const facts = logFacts.get(job.job_id);
          const started = job.started_at ? Date.parse(job.started_at) : NaN;
          if (["running", "stopping"].includes(job.status) && Number.isFinite(started)) {
            return Math.max(0, Math.floor((Date.now() - started) / 1000));
          }
          const finished = job.finished_at ? Date.parse(job.finished_at) : NaN;
          if (Number.isFinite(started) && Number.isFinite(finished) && finished >= started) {
            return Math.floor((finished - started) / 1000);
          }
          if (Number.isFinite(facts?.begin) && Number.isFinite(facts?.end) && facts.end >= facts.begin) {
            return Math.floor((facts.end - facts.begin) / 1000);
          }
          return null;
        }

        function renderProgress(job, fallback = "진행 중인 학습이 없습니다.") {
          const facts = job ? logFacts.get(job.job_id) : null;
          const percent = job?.status === "running" ? facts?.percent : null;
          progressBar.hidden = !job;
          if (Number.isInteger(percent)) progressBar.value = percent;
          else progressBar.removeAttribute("value");
          progressPercent.textContent = Number.isInteger(percent) ? `${percent}%` : "";
          const phase = job?.status === "stopping" ? "학습 중단 중" : facts?.phase ?? "학습 진행 확인 중";
          const elapsed = job ? duration(jobDuration(job)) : "—";
          progressPhase.textContent = job
            ? `${models[job.model] ?? job.model} · ${phase}${elapsed === "—" ? "" : " · " + elapsed}`
            : fallback;
        }

        function registration(job) {
          const labels = {
            not_requested: "등록 안 함", pending: "등록 대기",
            failed: "등록 실패", unconfirmed: "확인 안 됨",
          };
          return job.registered_model_version
            ? `v${job.registered_model_version}`
            : labels[job.registration_status] ?? "기록 없음";
        }

        function updateButtons() {
          start.disabled = !ready || busy || Boolean(activeJob);
          stop.disabled = !ready || busy || activeJob?.status !== "running";
          retry.disabled = start.disabled;
        }

        function notice(message, isError = false) {
          feedback.textContent = message;
          feedback.classList.toggle("error", isError);
        }

        async function api(suffix, options) {
          const response = await fetch("/api/training/jobs" + suffix, {
            credentials: "same-origin", cache: "no-store", ...options,
          });
          if (!response.ok) {
            const messages = {
              401: "세션이 만료되었습니다. 다시 로그인하세요.",
              403: "관리자 권한이 필요합니다.",
              404: "재학습 서비스 또는 요청한 작업을 찾을 수 없습니다.",
              409: "다른 학습이 실행 중이거나 작업 상태가 바뀌었습니다.",
              422: "입력한 학습 설정을 확인하세요.",
              502: "GPU 학습 서비스에서 응답을 받지 못했습니다.",
              503: "GPU 학습 서비스를 사용할 수 없습니다.",
            };
            throw new Error(messages[response.status] || "재학습 요청을 처리하지 못했습니다.");
          }
          return response.json();
        }

        function render(jobs) {
          activeJob = jobs.find(job => ["running", "stopping"].includes(job.status)) || null;
          selectedJob = jobs.find(job => job.job_id === picked) || activeJob || jobs[0] || null;
          current.textContent = activeJob
            ? `${models[activeJob.model] ?? activeJob.model} · ${states[activeJob.status]}`
            : "진행 중인 학습이 없습니다.";
          empty.textContent = jobs.length ? "" : "아직 학습 이력이 없습니다.";
          renderProgress(activeJob);

          const focusedJob = history.contains(document.activeElement)
            ? document.activeElement.dataset.trainingJob : null;
          let focusTarget = null;
          const rows = jobs.map(job => {
            const row = document.createElement("tr");
            const selected = job.job_id === selectedJob?.job_id;
            row.dataset.selected = String(selected);
            const started = job.started_at ? new Date(job.started_at) : null;
            row.insertCell().textContent = started && !Number.isNaN(started.getTime())
              ? dates.format(started) : "—";
            const button = document.createElement("button");
            button.type = "button";
            button.className = "history-view-button";
            button.textContent = models[job.model] ?? job.model;
            button.dataset.trainingJob = job.job_id;
            button.setAttribute("aria-pressed", String(selected));
            button.setAttribute("aria-label", `${button.textContent} 작업 상세 · ${job.job_id}`);
            if (focusedJob === job.job_id) focusTarget = button;
            row.insertCell().append(button);
            const values = [
              states[job.status] ?? "상태 확인 필요",
              duration(jobDuration(job)),
              job.model === "anomaly" ? "해당 없음" : job.fold ?? logFacts.get(job.job_id)?.fold ?? "—",
              registration(job),
            ];
            for (const value of values) row.insertCell().textContent = value;
            const run = row.insertCell();
            run.className = "training-run-id";
            run.textContent = job.run_id || logFacts.get(job.job_id)?.runId || "—";
            row.addEventListener("click", () => {
              picked = job.job_id;
              refresh();
            });
            return row;
          });
          history.replaceChildren(...rows);
          focusTarget?.focus({preventScroll: true});

          if (selectedJob) {
            const fields = [models[selectedJob.model] ?? selectedJob.model,
              states[selectedJob.status] ?? "상태 확인 필요", duration(jobDuration(selectedJob))];
            if (selectedJob.registered_model_name) {
              fields.push(`${selectedJob.registered_model_name} ${registration(selectedJob)}`);
            }
            if (selectedJob.batch_id) {
              const waiting = jobs.some(job => job.batch_id === selectedJob.batch_id && job.status === "queued");
              if (waiting) fields.push("다음 모델 대기 중");
            }
            detail.textContent = fields.join(" · ");
            retry.hidden = !["cancelled", "failed", "busy"].includes(selectedJob.status)
              || typeof selectedJob.register_model !== "boolean";
          } else {
            detail.textContent = "선택한 작업이 없습니다.";
            retry.hidden = true;
          }
          updateButtons();
        }

        async function refresh() {
          clearTimeout(timer);
          request?.abort();
          if (!visible || busy) return;
          const control = new AbortController();
          request = control;
          const options = { signal: control.signal };
          try {
            const jobs = await api("", options);
            if (control.signal.aborted) return;
            ready = true;
            const jobIds = new Set(jobs.map(job => job.job_id));
            for (const id of logFacts.keys()) if (!jobIds.has(id)) logFacts.delete(id);
            render(jobs);
            const selected = selectedJob;
            if (!selected) {
              logTitle.textContent = "로그";
              log.textContent = "";
              displayedLogJob = null;
              previousLogJob = null;
              return;
            }
            logTitle.textContent = `로그 · ${selected.job_id}`;
            if (displayedLogJob !== selected.job_id) {
              log.textContent = "로그를 불러오는 중입니다.";
              displayedLogJob = selected.job_id;
              previousLogJob = null;
            }
            const targets = new Map();
            if (previousLogJob !== selected.job_id) targets.set(selected.job_id, selected);
            if (activeJob) targets.set(activeJob.job_id, activeJob);
            if (!targets.size) return;

            await Promise.all([...targets.values()].map(async (job) => {
              try {
                const result = await api(`/${encodeURIComponent(job.job_id)}/log`, options);
                if (control.signal.aborted) return;
                if (typeof result.log === "string" && result.log) {
                  logFacts.set(job.job_id, {...logFacts.get(job.job_id), ...readLogFacts(result.log)});
                }
                if (job.job_id === selected.job_id) {
                  log.textContent = result.log || "아직 기록된 로그가 없습니다.";
                  if (!["queued", "running", "stopping"].includes(job.status)) {
                    previousLogJob = job.job_id;
                  }
                }
              } catch (error) {
                if (!control.signal.aborted && job.job_id === selected.job_id) log.textContent = error.message;
              }
            }));
            if (!control.signal.aborted) render(jobs);
          } catch (error) {
            if (control.signal.aborted) return;
            ready = false;
            activeJob = null;
            selectedJob = null;
            displayedLogJob = null;
            previousLogJob = null;
            history.replaceChildren();
            detail.textContent = "학습 상세를 불러오지 못했습니다.";
            retry.hidden = true;
            log.textContent = "";
            logTitle.textContent = "로그";
            current.textContent = error.message;
            renderProgress(null, "학습 상태를 불러오지 못했습니다.");
            empty.textContent = "학습 이력을 불러오지 못했습니다.";
            updateButtons();
          } finally {
            if (!control.signal.aborted && visible) timer = setTimeout(refresh, 3000);
          }
        }

        async function change(suffix, payload, success) {
          if (!visible || !ready || busy) return;
          const currentSession = session;
          busy = true;
          request?.abort();
          clearTimeout(timer);
          notice("");
          updateButtons();
          try {
            const job = await api(suffix, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(payload),
            });
            if (currentSession === session) {
              picked = job.batch_id ? null : job.job_id;
              notice(success);
            }
          } catch (error) {
            if (currentSession === session) notice(error.message, true);
          } finally {
            if (currentSession === session) {
              busy = false;
              updateButtons();
              refresh();
            }
          }
        }

        model.addEventListener("change", () => {
          const classifier = model.value !== "anomaly";
          fold.disabled = !classifier;
          element("fold-field").hidden = !classifier;
        });
        form.addEventListener("submit", (event) => {
          event.preventDefault();
          if (start.disabled) return;
          const payload = { model: model.value, register_model: register.checked };
          if (!fold.disabled && fold.value !== "") payload.fold = Number(fold.value);
          change("", payload, "학습 시작 요청을 접수했습니다.");
        });
        retry.addEventListener("click", () => {
          if (retry.hidden || retry.disabled || !selectedJob) return;
          model.value = selectedJob.model;
          model.dispatchEvent(new Event("change"));
          fold.value = selectedJob.fold ?? 0;
          register.checked = selectedJob.register_model;
          notice("설정을 불러왔습니다. 검토한 뒤 학습 시작을 누르세요.");
          start.focus();
        });
        stop.addEventListener("click", () => {
          if (!stop.disabled && activeJob) {
            change(`/${encodeURIComponent(activeJob.job_id)}/stop`, {}, "학습 중단을 요청했습니다.");
          }
        });

        return {
          setVisible(value) {
            visible = value;
            ready = false;
            updateButtons();
            if (value) {
              refresh();
              return;
            }
            request?.abort();
            clearTimeout(timer);
          },
          reset() {
            session++;
            this.setVisible(false);
            ready = false;
            busy = false;
            activeJob = null;
            picked = null;
            selectedJob = null;
            displayedLogJob = null;
            previousLogJob = null;
            retry.hidden = true;
            detail.textContent = "선택한 작업이 없습니다.";
            register.checked = false;
            history.replaceChildren();
            current.textContent = "학습 상태를 불러오는 중입니다.";
            logFacts.clear();
            renderProgress(null, "학습 상태를 확인하는 중입니다.");
            empty.textContent = "학습 이력을 불러오는 중입니다.";
            logTitle.textContent = "로그";
            log.textContent = "";
            notice("");
            updateButtons();
          },
        };
      })();

      function resetMonitoring() {
        selectedMonitorView = "worker";
        for (const { panel, frames } of monitorViews) {
          panel.hidden = true;
          for (const frame of frames) frame.removeAttribute("src");
        }
      }

      function showMonitorView(name) {
        if (monitorPanel.hidden || !canOpenTab("monitor")) return;
        if (name === "admin" && currentRole !== "admin") return;

        selectedMonitorView = name;
        for (const { button, panel, frames } of monitorViews) {
          const active = button.dataset.monitorView === name;
          button.setAttribute("aria-pressed", String(active));
          panel.hidden = !active;
          if (!active) continue;
          for (const frame of frames) {
            if (!frame.hasAttribute("src")) frame.src = frame.dataset.src;
          }
        }
      }

      for (const { button } of monitorViews) {
        button.addEventListener("click", () => showMonitorView(button.dataset.monitorView));
      }

      function showTab(name) {
        if (!canOpenTab(name)) return;
        for (const button of tabs) {
          const active = button.dataset.tab === name;
          document.getElementById(`panel-${button.dataset.tab}`).hidden = !active;
          button.setAttribute("aria-pressed", String(active));
        }
        if (name === "monitor") showMonitorView(selectedMonitorView);
        training.setVisible(name === "training");
        if (name === "history" && currentRole === "admin") loadHistoryWorkers();
      }
      for (const button of tabs) {
        button.addEventListener("click", () => showTab(button.dataset.tab));
      }
      async function refreshSession() {
        const response = await fetch("/api/auth/me");
        if (response.ok) {
          const user = await response.json();
          const accountName = document.getElementById("account-name");
          const account = user.account_name || user.email || "";
          accountName.textContent = account.split("@", 1)[0];
          accountName.hidden = !accountName.textContent;
          document.getElementById("logout-button").hidden = false;
          const roleChanged = currentRole !== user.role;
          currentRole = user.role;
          if (roleChanged) {
            resetHistory();
            training.reset();
            resetMonitoring();
          }
          monitoringViews.hidden = currentRole !== "admin";
          for (const button of tabs) button.hidden = !canOpenTab(button.dataset.tab);
          document.getElementById("history-title").textContent = user.role === "admin" ? "전체 검사 이력" : "내 검사 이력";
          showTab("inspect");
          inviteForm.hidden = user.role !== "admin";
          form.hidden = true;
          registerForm.hidden = true;
          screen.style.removeProperty("display");
          window.dispatchEvent(new Event("session-ready"));
          return;
        }
        if (response.status !== 401) {
          throw new Error("로그인 상태를 확인할 수 없습니다.");
        }
        const accountName = document.getElementById("account-name");
        currentRole = null;
        training.reset();
        resetMonitoring();
        monitoringViews.hidden = true;
        resetHistory();
        accountName.textContent = "";
        accountName.hidden = true;
        document.getElementById("logout-button").hidden = true;
        inviteForm.hidden = true;
        form.hidden = Boolean(inviteToken);
        registerForm.hidden = !inviteToken;
        screen.style.display = "none";
      }

      window.addEventListener("inspection-session-expired", () => {
        refreshSession().then(() => {
          if (!form.hidden) status.textContent = "세션이 만료되었습니다. 다시 로그인한 뒤 검사 재개를 누르세요.";
        }).catch((error) => {
          status.textContent = error.message;
        });
      });

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        submit.disabled = true;
        status.textContent = "";

        try {
          const response = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              email: form.elements.email.value.trim(),
              password: form.elements.password.value,
            }),
          });

          if (response.status === 401) {
            status.textContent = "이메일 또는 비밀번호가 올바르지 않습니다.";
            return;
          }
          if (!response.ok) {
            throw new Error("로그인 요청에 실패했습니다.");
          }

          form.reset();
          await refreshSession();
          if (!form.hidden) {
            status.textContent = "세션을 확인할 수 없습니다. 접속 주소가 HTTPS인지 확인하세요.";
          }
        } catch (error) {
          status.textContent = error.message;
        } finally {
          submit.disabled = false;
        }
      });
      inviteForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = inviteForm.querySelector('button[type="submit"]');
        const link = document.getElementById("invite-link");
        const status = document.getElementById("invite-status");
        button.disabled = true;
        link.value = "";
        status.textContent = "";

        try {
          const response = await fetch("/api/auth/invitations", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: inviteForm.elements.email.value.trim() }),
          });
          if (response.status !== 201) {
            throw new Error(`초대 발급 실패 (${response.status})`);
          }

          const { invite_token } = await response.json();
          const url = new URL(window.location.pathname, window.location.origin);
          url.hash = new URLSearchParams({ invite: invite_token }).toString();
          link.value = url.href;
          status.textContent = "초대 링크가 발급됐습니다.";
        } catch (error) {
          status.textContent = error.message;
        } finally {
          button.disabled = false;
        }
      });
      const logoutButton = document.getElementById("logout-button");
      const logoutStatus = document.getElementById("logout-status");

      logoutButton.addEventListener("click", async () => {
        logoutButton.disabled = true;
        logoutStatus.textContent = "";

        try {
          const response = await fetch("/api/auth/logout", { method: "POST" });
          if (!response.ok) {
            throw new Error("로그아웃에 실패했습니다. 다시 시도하세요.");
          }
          window.location.reload();
        } catch (error) {
          logoutStatus.textContent = error.message;
          logoutButton.disabled = false;
        }
      });
      registerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = registerForm.querySelector('button[type="submit"]');
        button.disabled = true;
        registerStatus.textContent = "";

        try {
          const response = await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              invite_token: inviteToken,
              password: registerForm.elements.password.value,
            }),
          });

          if (response.status === 400) {
            throw new Error("유효하지 않거나 이미 사용된 초대입니다.");
          }
          if (response.status === 409) {
            throw new Error("이미 등록된 계정입니다.");
          }
          if (response.status !== 201) {
            throw new Error(`가입에 실패했습니다 (${response.status}).`);
          }

          inviteToken = null;
          window.history.replaceState(
            null, "", window.location.pathname + window.location.search,
          );
          registerForm.reset();
          registerForm.hidden = true;
          form.hidden = false;
          status.textContent = "가입이 완료됐습니다. 초대받은 이메일로 로그인하세요.";
        } catch (error) {
          registerStatus.textContent = error.message;
        } finally {
          button.disabled = false;
        }
      });
      const historyButton = document.getElementById("load-history");
      const historyStatus = document.getElementById("history-status");
      const historyTable = document.getElementById("history-table");
      const historyRows = document.getElementById("history-rows");
      const historyDetail = document.getElementById("history-detail");
      const historyDetailSelection = document.getElementById("history-detail-selection");
      const historyDetailData = document.getElementById("history-detail-data");
      const stackedHistory = window.matchMedia("(max-width: 950px)");

      const historySearch = document.getElementById("history-search");
      const historyPeriod = document.getElementById("history-period");
      const scoreRanges = document.querySelectorAll(".history-score-range");
      for (const range of scoreRanges) {
        const track = range.querySelector(".history-score-track");
        const [lower, upper] = track.querySelectorAll('input[type="range"]');
        const output = range.querySelector("output");
        const minimum = Number(lower.min);
        const maximum = Number(upper.max);

        const render = () => {
          const start = lower.valueAsNumber;
          const end = upper.valueAsNumber;
          track.style.setProperty("--range-start", `${100 * (start - minimum) / (maximum - minimum)}%`);
          track.style.setProperty("--range-end", `${100 * (end - minimum) / (maximum - minimum)}%`);
          track.dataset.atMax = String(start === maximum);
          output.value = `${start.toFixed(4)} ~ ${end.toFixed(4)}`;
          lower.setAttribute("aria-valuetext", start.toFixed(4));
          upper.setAttribute("aria-valuetext", end.toFixed(4));
        };
        lower.addEventListener("input", () => {
          lower.value = Math.min(lower.valueAsNumber, upper.valueAsNumber);
          render();
        });
        upper.addEventListener("input", () => {
          upper.value = Math.max(upper.valueAsNumber, lower.valueAsNumber);
          render();
        });
        track.addEventListener("pointerdown", event => {
          if (event.target !== track || event.button !== 0) return;
          const {left, width} = track.getBoundingClientRect();
          const ratio = Math.max(0, Math.min(1, (event.clientX - left - 8) / Math.max(1, width - 16)));
          const value = minimum + ratio * (maximum - minimum);
          const slider = value <= (lower.valueAsNumber + upper.valueAsNumber) / 2 ? lower : upper;
          const step = Number(slider.step);
          slider.value = (minimum + Math.round((value - minimum) / step) * step).toFixed(4);
          slider.dispatchEvent(new Event("input"));
          slider.focus({preventScroll:true});
        });
        render();
      }
      const pagination = document.getElementById("history-pagination");
      const workerField = document.getElementById("history-worker-field");
      const workerSelect = document.getElementById("history-worker-select");
      let historySession = 0;
      let workerListState = "idle";
      const historyDate = new Intl.DateTimeFormat("ko-KR", {
        dateStyle: "short", timeStyle: "medium", timeZone: "Asia/Seoul", hourCycle: "h23",
      });
      const defectNames = { brown: "갈적색", blue: "청색", black: "흑색", surface: "표면" };
      let historyPage = 1;
      let historyFilters = new URLSearchParams();
      let activeHistoryButton;

      for (const select of historySearch.querySelectorAll("[data-hour], [data-minute]")) {
        const count = select.hasAttribute("data-hour") ? 24 : 60;
        for (let number = 0; number < count; number++) {
          const value = String(number).padStart(2, "0");
          select.add(new Option(value, value));
        }
      }

      function resetHistory() {
        ++historySession;
        workerListState = "idle";
        workerField.hidden = true;
        workerSelect.disabled = true;
        workerSelect.replaceChildren(new Option("전체 작업자", ""));
        historyPage = 1;
        historyFilters = new URLSearchParams();
        historyRows.replaceChildren();
        historyTable.hidden = true;
        historyDetailSelection.textContent = "검사 기록의 행을 선택하세요.";
        historyDetailData.replaceChildren();
        historyDetailData.hidden = true;
        pagination.replaceChildren();
        pagination.hidden = true;
        activeHistoryButton = undefined;
        historyStatus.textContent = "";
        historyButton.disabled = false;
      }

      async function loadHistoryWorkers() {
        if (currentRole !== "admin" || workerListState !== "idle") return;
        const session = historySession;
        workerListState = "loading";
        workerField.hidden = false;
        workerSelect.replaceChildren(new Option("작업자 불러오는 중", ""));
        try {
          const response = await fetch("/api/inference/history/workers", {cache: "no-store"});
          if (!response.ok) throw new Error("작업자 목록 조회 실패");
          const {workers} = await response.json();
          if (!Array.isArray(workers)) throw new Error("작업자 목록 형식 오류");
          if (session !== historySession) return;
          const names = new Map();
          for (const {worker_account} of workers) {
            names.set(worker_account, (names.get(worker_account) ?? 0) + 1);
          }
          workerSelect.replaceChildren(new Option("전체 작업자", ""), ...workers.map(
            ({worker_id, worker_account}) => new Option(
              names.get(worker_account) > 1 ? `${worker_account} (#${worker_id})` : worker_account,
              String(worker_id)
            )
          ));
          workerSelect.disabled = false;
          workerListState = "loaded";
        } catch {
          if (session !== historySession) return;
          workerSelect.replaceChildren(new Option("목록 조회 실패 · 이력 탭 다시 열기", ""));
          workerListState = "idle";
        }
      }

      function addHistoryRecord(record) {
        const row = historyRows.insertRow();
        const values = [
          record.worker_account || "계정명 미기록",
          record.detector_version == null ? "—" : `v${record.detector_version}`,
          record.classifier_version == null ? "—" : `v${record.classifier_version}`,
          record.prediction === "normal" ? "정상" : "불량",
          defectNames[record.defect_type] ?? (record.prediction === "normal" ? "해당 없음" : "분류 결과 없음"),
          historyDate.format(new Date(record.created_at)),
        ];
        const filenameCell = row.insertCell();
        for (const value of values) row.insertCell().textContent = value;
        row.cells[4].dataset.verdict = record.prediction === "normal" ? "normal"
          : defectNames[record.defect_type] ? "defect" : "untyped";

        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-view-button";
        button.textContent = record.filename;
        button.setAttribute("aria-label", `${record.filename} 상세 보기`);
        button.setAttribute("aria-controls", "history-detail");
        button.setAttribute("aria-pressed", "false");
        row.addEventListener("click", event => {
          const selection = window.getSelection();
          if (event.target !== button && selection && !selection.isCollapsed &&
              row.contains(selection.anchorNode)) return;
          activeHistoryButton?.setAttribute("aria-pressed", "false");
          activeHistoryButton = button;
          button.setAttribute("aria-pressed", "true");
          showHistoryDetail(record);
        });
        filenameCell.append(button);
      }

      function historyCard(title, fields) {
        const card = document.createElement("section");
        card.className = "history-detail-card";
        const heading = document.createElement("h4");
        heading.textContent = title;
        card.append(heading);
        if (fields.length) {
          const list = document.createElement("dl");
          for (const [label, value] of fields) {
            const row = document.createElement("div");
            const term = document.createElement("dt");
            const detail = document.createElement("dd");
            term.textContent = label;
            detail.textContent = value ?? "—";
            row.append(term, detail);
            list.append(row);
          }
          card.append(list);
        }
        return card;
      }

      function showHistoryDetail(record) {
        const duration = (value) => value == null ? "—" : `${value} ms`;
        const score = (value) => {
          if (typeof value === "string" && value.trim()) value = Number(value);
          return Number.isFinite(value) ? value.toFixed(4) : "미기록";
        };
        const storedRegions = Array.isArray(record.regions) ? record.regions : [];
        const regionScores = storedRegions.map((region, index) => [
          `영역 ${index + 1} 분류 점수`, score(region?.score),
        ]);
        const defect = record.is_defect == null ? "탐지 결과 없음"
          : record.is_defect ? "탐지 결함 있음" : "탐지 결함 없음";
        historyDetailSelection.textContent = `${record.filename} · ${defect}`;
        const models = document.createElement("div");
        models.className = "history-detail-grid";
        models.append(
          historyCard("이상치 탐지", [
            ["이상 점수", score(record.anomaly_score)],
            ["판정 기준", score(record.detector_threshold)],
            ["모델 URI", record.detector_name],
          ]),
          historyCard("결함 분류", [
            ...(regionScores.length ? regionScores : [["대표 분류 점수", score(record.confidence)]]),
            ["모델 URI", record.classifier_name],
          ])
        );

        const regions = historyCard("결함 위치", []);
        if (storedRegions.length) {
          const table = document.createElement("table");
          table.setAttribute("aria-label", "결함 위치");
          const header = table.createTHead().insertRow();
          for (const label of ["영역", "바운딩 박스 좌표"]) {
            const th = document.createElement("th");
            th.scope = "col";
            th.textContent = label;
            header.append(th);
          }
          const body = table.createTBody();
          storedRegions.forEach((region, index) => {
            const box = Array.isArray(region) ? region : region?.bbox;
            const row = body.insertRow();
            row.insertCell().textContent = String(index + 1);
            row.insertCell().textContent = formatInspectionBox(box);
          });
          regions.append(table);
        } else {
          const empty = document.createElement("p");
          empty.textContent = record.regions == null ? "기록된 결함 위치 없음" : "표시할 결함 영역 없음";
          regions.append(empty);
        }

        historyDetailData.replaceChildren(
          models,
          regions,
          historyCard("처리 시간", [
            ["전체", duration(record.total_latency_ms)],
            ["탐지", duration(record.detector_latency_ms)],
            ["분류", duration(record.classifier_latency_ms)],
          ]),
          historyCard("기록 및 검토", [
            ["요청 ID", record.id],
            ["모델 판정", ({normal: "정상", defect: "불량"})[record.model_prediction ?? record.prediction] ?? "—"],
            ["모델 분류 유형", defectNames[record.model_defect_type] ?? "—"],
            ["최종 판정", ({normal: "최종 정상", defect: "최종 불량"})[record.reviewed_label] ?? "작업자 판정 없음"],
            ["작업자 결함 유형", defectNames[record.reviewed_defect_type] ?? null],
            ["최종 판정 시간", record.reviewed_at ? historyDate.format(new Date(record.reviewed_at)) : null],
          ]),
          ...(record.reviewed_boxes?.length ? [
            historyCard("작업자 지정 결함 위치", record.reviewed_boxes.map(
              (box, index) => [`위치 ${index + 1}`, formatInspectionBox(box)]
            ))
          ] : [])
        );
        historyDetailData.hidden = false;
        if (stackedHistory.matches) historyDetail.focus();
      }

      function renderHistoryPages(page, totalPages) {
        pagination.replaceChildren();
        pagination.hidden = totalPages === 0;
        if (!totalPages) return;

        function addButton(label, target, accessibleLabel, disabled = false, current = false) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = label;
          button.setAttribute("aria-label", accessibleLabel);
          button.disabled = disabled;
          if (current) button.setAttribute("aria-current", "page");
          button.addEventListener("click", () => loadHistory(target));
          pagination.append(button);
        }
        addButton("<<", 1, "첫 페이지", page <= 1);
        addButton("‹", page - 1, "이전 페이지", page <= 1);
        const first = Math.floor((page - 1) / 10) * 10 + 1;
        for (let number = first; number <= Math.min(first + 9, totalPages); number++) {
          addButton(String(number), number, `${number}페이지`, number === page, number === page);
        }
        addButton("›", page + 1, "다음 페이지", page >= totalPages);
        addButton(">>", totalPages, "마지막 페이지", page >= totalPages);
      }

      async function loadHistory(page, filters = historyFilters) {
        if (historyButton.disabled) return;
        const session = historySession;
        historyButton.disabled = true;
        historyStatus.textContent = "검사 이력을 불러오는 중입니다.";
        try {
          const params = new URLSearchParams(filters);
          params.set("page", page);
          const response = await fetch(`/api/inference/history?${params}`);
          if (response.status === 401) throw new Error("다시 로그인해 주세요.");
          if (response.status === 422) throw new Error("검색 조건과 날짜 범위를 확인해 주세요.");
          if (!response.ok) throw new Error(`검사 이력 조회 실패 (${response.status})`);

          const { records, total, total_pages: totalPages } = await response.json();
          if (session !== historySession) return;
          if (!Number.isInteger(totalPages) || !Number.isInteger(total)) {
            throw new Error("페이지 정보를 반환하는 API 버전을 배포해 주세요.");
          }
          renderHistoryPages(page, totalPages);
          historyRows.replaceChildren();
          activeHistoryButton = undefined;
          historyDetailSelection.textContent = records.length
            ? "검사 기록의 행을 선택하세요."
            : "표시할 검사 기록이 없습니다.";
          historyDetailData.replaceChildren();
          historyDetailData.hidden = true;
          for (const record of records) addHistoryRecord(record);
          historyPage = page;
          historyFilters = filters;
          historyTable.hidden = records.length === 0;
          historyStatus.textContent = records.length
            ? `총 ${total.toLocaleString("ko-KR")}건 · ${page}/${totalPages}페이지 · 현재 ${records.length}건`
            : `총 ${total.toLocaleString("ko-KR")}건 · 표시할 검사 이력이 없습니다.`;
        } catch (error) {
          if (session === historySession) historyStatus.textContent = error.message;
        } finally {
          if (session === historySession) historyButton.disabled = false;
        }
      }

      historySearch.addEventListener("submit", (event) => {
        event.preventDefault();
        const formData = new FormData(historySearch);
        const filters = new URLSearchParams();
        const names = [
          "date_from", "date_to", "filename", "prediction", "worker_id", "reviewed",
        ];
        for (const name of names) {
          const value = formData.get(name)?.trim();
          if (value) filters.set(name, value);
        }

        for (const kind of ["anomaly", "confidence"]) {
          const lower = historySearch.elements[`${kind}_min`];
          const upper = historySearch.elements[`${kind}_max`];
          if (lower.valueAsNumber > Number(lower.min) || upper.valueAsNumber < Number(upper.max)) {
            filters.set(lower.name, lower.value);
            filters.set(upper.name, upper.value);
          }
        }

        for (const suffix of ["from", "to"]) {
          const date = filters.get(`date_${suffix}`);
          const hour = formData.get(`hour_${suffix}`);
          const minute = formData.get(`minute_${suffix}`);
          if ((hour || minute) && !date) {
            historyPeriod.open = true;
            historyStatus.textContent = "시각을 지정하려면 해당 날짜도 선택해 주세요.";
            return;
          }
          if (!hour && minute) {
            historyPeriod.open = true;
            historyStatus.textContent = "분을 지정하려면 시도 선택해 주세요.";
            return;
          }
          if (hour) filters.set(`time_${suffix}`, `${hour}:${minute || "00"}`);
        }

        const start = filters.get("date_from");
        const end = filters.get("date_to");
        if (start && end && `${start}T${filters.get("time_from") ?? "00:00"}`
          > `${end}T${filters.get("time_to") ?? "23:59"}`) {
          historyPeriod.open = true;
          historyStatus.textContent = "종료 일시를 시작 일시와 같거나 이후로 선택해 주세요.";
          return;
        }
        historyPeriod.open = false;
        loadHistory(1, filters);
      });
      refreshSession().catch((error) => {
        form.hidden = false;
        status.textContent = error.message;
      });
    })();
