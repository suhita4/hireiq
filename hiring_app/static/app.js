// HireIQ — frontend logic
// State, role-aware data flow, three-pane rendering, kanban drag-and-drop.

const FUNNEL_STAGES = ["Screened", "Interview Scheduled", "Offered", "Hired", "Rejected"];
const STAGE_KEYS = { "1": "Screened", "2": "Interview Scheduled", "3": "Offered", "4": "Hired", "5": "Rejected" };

const SAMPLE_JDS = [
  { title: "Data Scientist", text: "We are hiring a Data Scientist with 3+ years of experience in machine learning and statistical modeling. Must be proficient in Python, scikit-learn, pandas, and numpy. Experience with TensorFlow or PyTorch is a strong plus. The role involves building classification and regression models, performing feature engineering, and presenting insights to non-technical stakeholders." },
  { title: "Senior Java Developer", text: "Looking for a Senior Java Developer with 5+ years of experience building enterprise-grade applications. Strong knowledge of Spring Boot, REST APIs, and SQL databases required. Experience with microservices architecture, Docker, and Jenkins is expected. Bachelor's degree in Computer Science or related field." },
  { title: "DevOps Engineer", text: "Seeking a DevOps Engineer with 4+ years of experience in AWS, Docker, Kubernetes, and CI/CD pipelines. The candidate should be proficient in Terraform, Ansible, and Jenkins. Strong scripting skills in Bash or Python are required. Experience with Linux systems administration, networking, and monitoring tools is expected." },
  { title: "HR Manager", text: "We need an experienced HR Manager with 7+ years in talent acquisition, onboarding, payroll, and employee relations. The role requires strong knowledge of HRIS systems, performance management processes, and labour compliance. Excellent communication and leadership skills are essential." },
  { title: "Python Developer", text: "Hiring a Python Developer with 3+ years of experience in backend development. Proficiency in Flask or Django, REST APIs, and SQL/NoSQL databases is required. Experience with pandas, numpy, and data pipelines is a strong advantage." },
];

const state = {
  roles: {},
  activeRoleId: null,
  jdEditing: false,
  candidates: [],
  jobEntities: { SKILL: [], JOB_TITLE: [], EXPERIENCE: [] },
  selectedCandidateId: null,
  funnel: {},          // { candidate_id_string: stage } for active role
  skillGap: {},
  totalCandidates: 0,
  filter: "all",       // all | in_funnel | Hired | Rejected
  compareSelected: new Set(),
  drawerOpen: false,
  drawerTab: "coverage",
  skillGapChart: null,
};

// ─── Boot ───────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);

async function init() {
  attachStaticHandlers();
  await loadRoles();
  if (Object.keys(state.roles).length === 0) {
    await createRole({ name: "New role", jd_text: "" });
  } else {
    state.activeRoleId = Object.keys(state.roles)[0];
  }
  await hydrateActiveRole();
}

function attachStaticHandlers() {
  document.getElementById("new-role-btn").addEventListener("click", onNewRole);
  document.getElementById("delete-role-btn").addEventListener("click", onDeleteRole);
  document.getElementById("role-select").addEventListener("change", e => switchRole(e.target.value));
  document.getElementById("role-name-input").addEventListener("change", onRoleRename);

  document.getElementById("role-switch-btn").addEventListener("click", toggleRolePopover);
  document.getElementById("role-popover").addEventListener("click", onRolePopoverClick);
  document.addEventListener("click", onDocumentClickForPopover);

  document.getElementById("filter-bar").addEventListener("click", onFilterChipClick);
  document.getElementById("compare-bar-btn").addEventListener("click", showCompareModal);
  document.getElementById("compare-modal-close").addEventListener("click", hideCompareModal);

  document.getElementById("drawer-toggle").addEventListener("click", toggleDrawer);
  document.querySelectorAll(".drawer-tab").forEach(tab =>
    tab.addEventListener("click", () => switchDrawerTab(tab.dataset.tab))
  );

  document.getElementById("help-btn").addEventListener("click", showHelp);
  document.getElementById("help-modal-close").addEventListener("click", hideHelp);
  document.getElementById("detail-close-btn").addEventListener("click", () => selectCandidate(null));

  document.addEventListener("keydown", onGlobalKeydown);
}

// ─── Roles API ─────────────────────────────────────────
async function loadRoles() {
  const res = await fetch("/roles");
  state.roles = await res.json();
}

async function createRole({ name = "New role", jd_text = "" } = {}) {
  const res = await fetch("/roles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, jd_text }),
  });
  const role = await res.json();
  state.roles[role.id] = { name: role.name, jd_text: role.jd_text, created_at: role.created_at };
  state.activeRoleId = role.id;
  return role;
}

async function patchRole(id, patch) {
  const res = await fetch(`/roles/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) { showToast("Could not save role."); return; }
  const role = await res.json();
  state.roles[id] = { name: role.name, jd_text: role.jd_text, created_at: role.created_at };
}

async function deleteRole(id) {
  await fetch(`/roles/${id}`, { method: "DELETE" });
  delete state.roles[id];
}

async function onNewRole() {
  await createRole({ name: "New role", jd_text: "" });
  state.candidates = [];
  state.selectedCandidateId = null;
  state.funnel = {};
  state.jdEditing = true;
  renderAll();
}

async function onDeleteRole() {
  if (!state.activeRoleId) return;
  const remaining = Object.keys(state.roles).length - 1;
  if (remaining === 0) {
    if (!confirm("Delete the only role? A new empty role will be created.")) return;
  } else {
    if (!confirm(`Delete "${state.roles[state.activeRoleId].name}"? This also clears its funnel.`)) return;
  }
  await deleteRole(state.activeRoleId);
  if (Object.keys(state.roles).length === 0) {
    await createRole({ name: "New role" });
  } else {
    state.activeRoleId = Object.keys(state.roles)[0];
  }
  state.candidates = [];
  state.selectedCandidateId = null;
  await hydrateActiveRole();
}

async function onRoleRename(e) {
  if (!state.activeRoleId) return;
  const newName = e.target.value.trim() || "Untitled Role";
  await patchRole(state.activeRoleId, { name: newName });
  renderRoleSwitcher();
}

async function switchRole(id) {
  state.activeRoleId = id;
  state.candidates = [];
  state.selectedCandidateId = null;
  state.compareSelected.clear();
  state.jdEditing = false;
  await hydrateActiveRole();
}

async function hydrateActiveRole() {
  renderRoleSwitcher();
  await loadFunnelForActiveRole();
  const role = state.roles[state.activeRoleId];
  if (role && role.jd_text && role.jd_text.length >= 20) {
    await runMatchAndAnalyze();
  } else {
    state.jdEditing = true;
  }
  renderAll();
}

// ─── Match & analyze ───────────────────────────────────
async function runMatchAndAnalyze() {
  const role = state.roles[state.activeRoleId];
  if (!role || !role.jd_text || role.jd_text.length < 20) return;

  document.getElementById("candidate-list").innerHTML =
    '<div class="loading"><div class="spinner"></div>Matching candidates…</div>';

  try {
    const [recRes, anRes] = await Promise.all([
      fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role_id: state.activeRoleId, top_n: 20 }),
      }),
      fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role_id: state.activeRoleId }),
      }),
    ]);
    if (!recRes.ok) {
      const err = await recRes.json().catch(() => ({}));
      showToast(err.error || "Could not load matches.");
      state.candidates = [];
    } else {
      const data = await recRes.json();
      state.candidates = data.candidates || [];
      state.jobEntities = data.job_entities || { SKILL: [], JOB_TITLE: [], EXPERIENCE: [] };
      state.totalCandidates = data.total_candidates_searched || 0;
    }
    if (anRes.ok) {
      const an = await anRes.json();
      state.skillGap = an.skill_gap || {};
    }
  } catch (e) {
    showToast("Could not connect to the server.");
    state.candidates = [];
  }
}

async function loadFunnelForActiveRole() {
  if (!state.activeRoleId) { state.funnel = {}; return; }
  const res = await fetch(`/funnel?role_id=${state.activeRoleId}`);
  state.funnel = res.ok ? await res.json() : {};
}

async function updateFunnelStage(candidateId, stage) {
  if (!state.activeRoleId) return;
  const res = await fetch("/funnel/update", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role_id: state.activeRoleId, id: candidateId, stage }),
  });
  if (!res.ok) { showToast("Could not save funnel stage."); return; }
  const cid = String(candidateId);
  if (stage) state.funnel[cid] = stage;
  else delete state.funnel[cid];
  renderCandidates();
  renderFunnelSummary();
  renderFunnelBoard();
  if (state.selectedCandidateId === candidateId) renderDetail();
}

// ─── JD pane ───────────────────────────────────────────
function renderJdPane() {
  const role = state.roles[state.activeRoleId];
  const body = document.getElementById("jd-pane-body");
  if (!role) {
    body.innerHTML = '<div class="jd-empty">Loading…</div>';
    return;
  }

  if (state.jdEditing || !role.jd_text) {
    const charCount = (role.jd_text || "").length;
    const draftHtml = role.jd_text ? escapeHtml(role.jd_text) : "";
    body.innerHTML = `
      <div class="jd-edit-block">
        <textarea id="jd-textarea" placeholder="Paste the job description here…">${draftHtml}</textarea>
        <div class="jd-meta" id="jd-meta">
          <span id="jd-charcount">${charCount} / 20 min</span>
          <span>Ctrl+Enter to save</span>
        </div>
        <div class="jd-actions">
          <button class="btn" id="jd-save-btn" ${charCount < 20 ? "disabled" : ""}>Save &amp; Match</button>
          ${role.jd_text ? '<button class="btn btn-ghost" id="jd-cancel-btn">Cancel</button>' : ""}
        </div>
        ${!role.jd_text ? renderSampleLinks() : ""}
      </div>
    `;

    const ta = document.getElementById("jd-textarea");
    const meta = document.getElementById("jd-meta");
    const cc = document.getElementById("jd-charcount");
    const saveBtn = document.getElementById("jd-save-btn");

    ta.addEventListener("input", () => {
      const len = ta.value.trim().length;
      cc.textContent = `${len} / 20 min`;
      meta.classList.toggle("invalid", len > 0 && len < 20);
      saveBtn.disabled = len < 20;
    });
    ta.addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !saveBtn.disabled) {
        e.preventDefault();
        onJdSave(ta.value);
      }
    });
    saveBtn.addEventListener("click", () => onJdSave(ta.value));

    const cancelBtn = document.getElementById("jd-cancel-btn");
    if (cancelBtn) cancelBtn.addEventListener("click", () => { state.jdEditing = false; renderJdPane(); });

    document.querySelectorAll("[data-sample]").forEach(a => {
      a.addEventListener("click", e => {
        e.preventDefault();
        ta.value = SAMPLE_JDS[Number(a.dataset.sample)].text;
        ta.dispatchEvent(new Event("input"));
        ta.focus();
      });
    });
  } else {
    const skills = state.jobEntities.SKILL || [];
    const titles = state.jobEntities.JOB_TITLE || [];
    body.innerHTML = `
      <div class="jd-summary">
        ${titles.length ? `
          <div>
            <div class="jd-section-label">Role</div>
            <div class="jd-skill-list">${titles.map(t => `<span class="tag title">${escapeHtml(t)}</span>`).join("")}</div>
          </div>` : ""}
        <div>
          <div class="jd-section-label">Required Skills</div>
          <div class="jd-skill-list">
            ${skills.length
              ? skills.map(s => `<span class="tag skill">${escapeHtml(s)}</span>`).join("")
              : '<span class="empty">None detected yet</span>'}
          </div>
        </div>
        <div class="jd-actions">
          <button class="btn btn-ghost" id="jd-edit-btn">Edit JD</button>
        </div>
      </div>
    `;
    document.getElementById("jd-edit-btn").addEventListener("click", () => {
      state.jdEditing = true;
      renderJdPane();
    });
  }
}

function renderSampleLinks() {
  return `
    <div style="margin-top:14px;font-size:12px;color:var(--muted);">
      Or load an example:
      <div class="examples">
        ${SAMPLE_JDS.map((s, i) => `<a data-sample="${i}" href="#">${s.title}</a>`).join("")}
      </div>
    </div>
  `;
}

async function onJdSave(text) {
  const trimmed = (text || "").trim();
  if (trimmed.length < 20) return;
  await patchRole(state.activeRoleId, { jd_text: trimmed });
  state.jdEditing = false;
  await runMatchAndAnalyze();
  renderAll();
}

// ─── Role switcher ─────────────────────────────────────
function renderRoleSwitcher() {
  const select = document.getElementById("role-select");
  const ids = Object.keys(state.roles);
  select.innerHTML = ids
    .map(id => `<option value="${id}" ${id === state.activeRoleId ? "selected" : ""}>${escapeHtml(state.roles[id].name)}</option>`)
    .join("");
  const role = state.roles[state.activeRoleId];
  document.getElementById("role-name-input").value = role ? role.name : "";
  renderRolePopover();
}

function renderRolePopover() {
  const popover = document.getElementById("role-popover");
  const ids = Object.keys(state.roles);
  if (ids.length === 0) {
    popover.innerHTML = `<div class="role-popover-empty">No roles yet.</div>`;
    return;
  }
  const checkSvg = `<svg class="check" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 12l5 5L20 7"/></svg>`;
  popover.innerHTML = ids.map(id => {
    const isActive = id === state.activeRoleId;
    return `<button type="button" role="option" aria-selected="${isActive}" class="role-popover-item${isActive ? " active" : ""}" data-role-id="${id}">
      <span class="role-popover-label">${escapeHtml(state.roles[id].name)}</span>
      ${checkSvg}
    </button>`;
  }).join("");
}

function toggleRolePopover(e) {
  e.stopPropagation();
  const btn = document.getElementById("role-switch-btn");
  const popover = document.getElementById("role-popover");
  const isOpen = !popover.hidden;
  if (isOpen) {
    closeRolePopover();
  } else {
    popover.hidden = false;
    btn.setAttribute("aria-expanded", "true");
  }
}

function closeRolePopover() {
  const popover = document.getElementById("role-popover");
  if (popover.hidden) return;
  popover.hidden = true;
  document.getElementById("role-switch-btn").setAttribute("aria-expanded", "false");
}

function onRolePopoverClick(e) {
  const item = e.target.closest(".role-popover-item");
  if (!item) return;
  const id = item.dataset.roleId;
  closeRolePopover();
  if (id && id !== state.activeRoleId) {
    const select = document.getElementById("role-select");
    select.value = id;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

function onDocumentClickForPopover(e) {
  const popover = document.getElementById("role-popover");
  if (popover.hidden) return;
  if (e.target.closest(".role-switcher")) return;
  closeRolePopover();
}

// ─── Funnel summary ────────────────────────────────────
function renderFunnelSummary() {
  const counts = { Hired: 0, "in pipeline": 0 };
  for (const stage of Object.values(state.funnel)) {
    if (stage === "Hired") counts.Hired += 1;
    else if (stage && stage !== "Rejected") counts["in pipeline"] += 1;
  }
  document.getElementById("funnel-summary").innerHTML = `
    <span><span class="dot" style="background:var(--green)"></span><strong>${counts.Hired}</strong> hired</span>
    <span><span class="dot" style="background:var(--accent)"></span><strong>${counts["in pipeline"]}</strong> in pipeline</span>
  `;
}

// ─── Candidates list ──────────────────────────────────
function renderCandidates() {
  const list = document.getElementById("candidate-list");
  let filtered = state.candidates;
  if (state.filter === "in_funnel") filtered = filtered.filter(c => state.funnel[String(c.id)]);
  else if (state.filter === "Hired") filtered = filtered.filter(c => state.funnel[String(c.id)] === "Hired");
  else if (state.filter === "Rejected") filtered = filtered.filter(c => state.funnel[String(c.id)] === "Rejected");

  if (state.candidates.length === 0) {
    list.innerHTML = '<div class="empty-list">No matches yet. Save a JD to see ranked candidates.</div>';
    return;
  }
  if (filtered.length === 0) {
    list.innerHTML = '<div class="empty-list">No candidates in this filter.</div>';
    return;
  }

  list.innerHTML = filtered.map((c, idx) => {
    const realIdx = state.candidates.indexOf(c);
    const rank = realIdx + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : "";
    const score = Math.round(c.total_score);
    const stage = state.funnel[String(c.id)] || "";
    const selected = state.selectedCandidateId === c.id ? "selected" : "";
    const compared = state.compareSelected.has(c.id) ? "compare-checked" : "";
    const topSkills = (c.matched_skills || []).slice(0, 3);
    const metaParts = [c.category, formatExperience(c.extracted_experience)]
      .filter(Boolean)
      .map(escapeHtml);
    const meta = metaParts.length ? `<div class="candidate-meta">${metaParts.join(" · ")}</div>` : "";
    const explanation = c.explanation
      ? `<div class="candidate-explanation" title="${escapeHtml(c.explanation)}">${escapeHtml(c.explanation)}</div>`
      : "";

    return `
      <div class="candidate-row ${selected} ${compared}" data-id="${c.id}">
        <input type="checkbox" class="compare-cb" data-id="${c.id}" ${state.compareSelected.has(c.id) ? "checked" : ""} title="Select for comparison">
        <div class="rank ${rankClass}">#${rank}</div>
        <div class="candidate-main">
          <div class="candidate-name">${escapeHtml(c.name || "Candidate #" + c.id)}</div>
          ${meta}
          <div class="candidate-skills-mini">
            ${topSkills.map(s => `<span class="skill-mini">${escapeHtml(s)}</span>`).join("")}
            ${c.matched_skills && c.matched_skills.length > 3
              ? `<span class="skill-mini">+${c.matched_skills.length - 3}</span>` : ""}
          </div>
          ${explanation}
        </div>
        <div class="score-cell">
          <div class="score-label">Match Score</div>
          <div class="score-pill">${score}</div>
        </div>
      </div>
    `;
  }).join("");

  list.querySelectorAll(".candidate-row").forEach(row => {
    row.addEventListener("click", e => {
      if (e.target.classList.contains("compare-cb")) return;
      selectCandidate(Number(row.dataset.id));
    });
  });
  list.querySelectorAll(".compare-cb").forEach(cb => {
    cb.addEventListener("click", e => e.stopPropagation());
    cb.addEventListener("change", e => toggleCompare(Number(e.target.dataset.id), e.target.checked));
  });

  renderCompareBar();
}

function selectCandidate(id) {
  state.selectedCandidateId = id;
  renderCandidates();
  renderDetail();
}

// ─── Detail pane ───────────────────────────────────────
function renderDetail() {
  const pane = document.getElementById("detail-body");
  const c = state.candidates.find(x => x.id === state.selectedCandidateId);
  document.querySelector(".main")?.classList.toggle("has-selection", !!c);
  if (!c) {
    pane.innerHTML = `
      <div class="detail-empty">
        <svg width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.4">
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 7a3 3 0 11-6 0 3 3 0 016 0zM4 20a8 8 0 0116 0"/>
        </svg>
        <div>Select a candidate to see details.</div>
      </div>
    `;
    return;
  }
  const stage = state.funnel[String(c.id)] || "";
  const score = Math.round(c.total_score);
  const matched = c.matched_skills || [];
  const missing = c.missing_skills || [];
  const stageOpts = ["", ...FUNNEL_STAGES]
    .map(s => `<option value="${s}" ${s === stage ? "selected" : ""}>${s || "Select stage"}</option>`)
    .join("");
  const sKeys = ["skill_score", "title_score", "experience_score", "semantic_score"];
  const sLabels = { skill_score: "Skills", title_score: "Job Title", experience_score: "Experience", semantic_score: "Semantic" };
  const sFill = { skill_score: "skill", title_score: "title", experience_score: "exp", semantic_score: "semantic" };
  const sMax = { skill_score: 60, title_score: 25, experience_score: 15, semantic_score: 100 };

  pane.innerHTML = `
    <div class="detail-head">
      <div>
        <div class="detail-name">${escapeHtml(c.name || "Candidate #" + c.id)}</div>
        <div class="detail-rank">Rank #${state.candidates.indexOf(c) + 1} · ${escapeHtml(c.category || "")}</div>
      </div>
      <div class="score-cell">
        <div class="score-label">Match Score</div>
        <div class="score-pill">${score}</div>
      </div>
    </div>

    ${c.explanation ? `<div class="detail-explanation">${escapeHtml(c.explanation)}</div>` : ""}

    <div class="detail-section">
      <div class="detail-section-label">Funnel stage</div>
      <div class="stage-select-row">
        <span class="stage-dot" data-stage="${stage}"></span>
        <select id="detail-stage-select">${stageOpts}</select>
      </div>
    </div>

    <div class="detail-section">
      <div class="detail-section-label">Matched skills (${matched.length})</div>
      <div class="detail-tags">
        ${matched.length
          ? matched.map(s => `<span class="tag skill">${escapeHtml(s)}</span>`).join("")
          : '<span class="tag empty">No skill overlap detected</span>'}
      </div>
    </div>

    ${missing.length ? `
      <div class="detail-section">
        <div class="detail-section-label">Missing skills (${missing.length})</div>
        <div class="detail-tags">
          ${missing.map(s => `<span class="tag missing">${escapeHtml(s)}</span>`).join("")}
        </div>
      </div>` : ""}

    ${c.resume_excerpt ? `
      <div class="detail-section">
        <div class="detail-section-label">Resume excerpt</div>
        <div class="resume-excerpt">${escapeHtml(c.resume_excerpt)}</div>
      </div>` : ""}

    <details class="disclosure detail-section">
      <summary>Why this score?</summary>
      <div class="breakdown">
        ${sKeys.map(k => {
          const v = c[k] || 0;
          const pct = Math.min(100, (v / sMax[k]) * 100);
          return `
            <div class="breakdown-row">
              <span class="breakdown-label">${sLabels[k]}</span>
              <div class="breakdown-track"><div class="breakdown-fill ${sFill[k]}" style="width:${pct}%"></div></div>
              <span class="breakdown-val">${Math.round(v)}</span>
            </div>`;
        }).join("")}
      </div>
    </details>
  `;
  document.getElementById("detail-stage-select").addEventListener("change", e =>
    updateFunnelStage(c.id, e.target.value)
  );
}

// ─── Filter chips ──────────────────────────────────────
function onFilterChipClick(e) {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  state.filter = chip.dataset.filter;
  document.querySelectorAll("#filter-bar .chip").forEach(c => c.classList.toggle("active", c === chip));
  renderCandidates();
}

// ─── Compare ───────────────────────────────────────────
function toggleCompare(id, checked) {
  if (checked) {
    if (state.compareSelected.size >= 2) {
      // Unchecking ourselves and warning is awkward — keep behavior of refusing the 3rd.
      const cb = document.querySelector(`.compare-cb[data-id="${id}"]`);
      if (cb) cb.checked = false;
      showToast("You can compare up to 2 candidates.");
      return;
    }
    state.compareSelected.add(id);
  } else {
    state.compareSelected.delete(id);
  }
  renderCandidates();
}

function renderCompareBar() {
  const bar = document.getElementById("compare-bar");
  const n = state.compareSelected.size;
  bar.style.display = n > 0 ? "flex" : "none";
  document.getElementById("compare-bar-count").textContent = `${n}/2 selected`;
  document.getElementById("compare-bar-btn").disabled = n !== 2;
}

function showCompareModal() {
  const ids = [...state.compareSelected];
  const cs = ids.map(id => state.candidates.find(c => c.id === id)).filter(Boolean);
  if (cs.length !== 2) return;
  const sLabels = { skill_score: "Skills", title_score: "Job Title", experience_score: "Experience", semantic_score: "Semantic" };
  const sFill = { skill_score: "skill", title_score: "title", experience_score: "exp", semantic_score: "semantic" };
  const sMax = { skill_score: 60, title_score: 25, experience_score: 15, semantic_score: 100 };
  document.getElementById("compare-grid").innerHTML = cs.map(c => `
    <div class="compare-col">
      <div class="compare-col-name">
        <span>${escapeHtml(c.name || "Candidate #" + c.id)}</span>
        <div class="score-cell">
          <div class="score-label">Match Score</div>
          <span class="score-pill">${Math.round(c.total_score)}</span>
        </div>
      </div>
      ${c.explanation ? `<div style="font-size:12px;color:var(--muted-strong);margin-bottom:10px">${escapeHtml(c.explanation)}</div>` : ""}
      <div class="breakdown">
        ${["skill_score","title_score","experience_score","semantic_score"].map(k => {
          const v = c[k] || 0;
          const pct = Math.min(100, (v / sMax[k]) * 100);
          return `
            <div class="breakdown-row">
              <span class="breakdown-label">${sLabels[k]}</span>
              <div class="breakdown-track"><div class="breakdown-fill ${sFill[k]}" style="width:${pct}%"></div></div>
              <span class="breakdown-val">${Math.round(v)}</span>
            </div>`;
        }).join("")}
      </div>
      <div style="margin-top:12px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Matched (${(c.matched_skills||[]).length})</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">
        ${(c.matched_skills||[]).map(s => `<span class="tag skill">${escapeHtml(s)}</span>`).join("") || '<span class="tag empty">None</span>'}
      </div>
      <div style="margin-top:10px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">Missing (${(c.missing_skills||[]).length})</div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">
        ${(c.missing_skills||[]).map(s => `<span class="tag missing">${escapeHtml(s)}</span>`).join("") || '<span class="tag empty">None</span>'}
      </div>
    </div>
  `).join("");
  document.getElementById("compare-overlay").classList.add("open");
}
function hideCompareModal() {
  document.getElementById("compare-overlay").classList.remove("open");
}

// ─── Drawer (skill coverage + funnel) ───────────────────
function toggleDrawer() {
  state.drawerOpen = !state.drawerOpen;
  document.getElementById("drawer").classList.toggle("collapsed", !state.drawerOpen);
}
function switchDrawerTab(tab) {
  state.drawerTab = tab;
  document.querySelectorAll(".drawer-tab").forEach(t => t.classList.toggle("active", t.dataset.tab === tab));
  document.querySelectorAll(".drawer-panel").forEach(p => p.classList.toggle("active", p.dataset.tab === tab));
  if (tab === "coverage") renderSkillCoverage();
  else renderFunnelBoard();
}

function renderSkillCoverage() {
  const blurb = document.getElementById("coverage-blurb");
  const entries = Object.entries(state.skillGap).sort((a, b) => a[1].pct_have - b[1].pct_have);
  if (!entries.length) {
    blurb.innerHTML = "Save a JD to see how its required skills are distributed across the candidate pool.";
    document.getElementById("coverage-canvas").style.display = "none";
    return;
  }
  document.getElementById("coverage-canvas").style.display = "block";

  const minSkill = entries[0];
  blurb.innerHTML = `
    Of <strong>${state.totalCandidates.toLocaleString()}</strong> candidates,
    only <strong>${minSkill[1].pct_have}%</strong> have <em>${escapeHtml(minSkill[0])}</em>.
    ${entries.filter(([, d]) => d.pct_have < 20).length} skills are hard to find — consider relaxing those.
  `;

  const labels = entries.map(([s]) => s);
  const values = entries.map(([, d]) => d.pct_have);
  const colors = values.map(v =>
    v >= 60 ? "rgba(45,212,160,0.7)" : v >= 30 ? "rgba(245,200,66,0.7)" : "rgba(239,68,68,0.7)"
  );

  if (state.skillGapChart) state.skillGapChart.destroy();
  state.skillGapChart = new Chart(document.getElementById("coverage-canvas"), {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }] },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${entries[ctx.dataIndex][1].pct_have}% (${entries[ctx.dataIndex][1].have}/${state.totalCandidates})` } },
      },
      scales: {
        x: { min: 0, max: 100, ticks: { color: "#9ca3af", callback: v => v + "%" }, grid: { color: "rgba(37,42,56,0.7)" } },
        y: { ticks: { color: "#e8eaf0" }, grid: { display: false } },
      },
    },
  });
}

function renderFunnelBoard() {
  const board = document.getElementById("funnel-board");
  // Build a name lookup from current candidates; fall back to id for funnel'd candidates not in current top-N.
  const nameById = new Map(state.candidates.map(c => [String(c.id), c.name || `Candidate #${c.id}`]));
  const stagedIds = Object.keys(state.funnel);

  board.innerHTML = FUNNEL_STAGES.map(stage => {
    const ids = stagedIds.filter(cid => state.funnel[cid] === stage);
    const items = ids.map(cid => `
      <div class="funnel-card" draggable="true" data-id="${cid}">
        ${escapeHtml(nameById.get(cid) || `Candidate #${cid}`)}
      </div>
    `).join("");
    return `
      <div class="funnel-col" data-stage="${stage}">
        <div class="funnel-col-head">
          <span class="funnel-col-name">${stage}</span>
          <span class="funnel-col-count">${ids.length}</span>
        </div>
        ${ids.length ? items : '<div class="funnel-empty">Drop candidates here</div>'}
      </div>
    `;
  }).join("");

  attachFunnelDragHandlers();
}

function attachFunnelDragHandlers() {
  let draggingId = null;
  document.querySelectorAll(".funnel-card").forEach(card => {
    card.addEventListener("dragstart", e => {
      draggingId = Number(card.dataset.id);
      card.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });
    card.addEventListener("dragend", () => card.classList.remove("dragging"));
  });
  document.querySelectorAll(".funnel-col").forEach(col => {
    col.addEventListener("dragover", e => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; col.classList.add("drag-over"); });
    col.addEventListener("dragleave", () => col.classList.remove("drag-over"));
    col.addEventListener("drop", e => {
      e.preventDefault();
      col.classList.remove("drag-over");
      if (draggingId == null) return;
      const stage = col.dataset.stage;
      updateFunnelStage(draggingId, stage);
      draggingId = null;
    });
  });
}

// ─── Help overlay ──────────────────────────────────────
function showHelp() { document.getElementById("help-overlay").classList.add("open"); }
function hideHelp() { document.getElementById("help-overlay").classList.remove("open"); }

// ─── Global keyboard ──────────────────────────────────
function onGlobalKeydown(e) {
  // Skip if typing in input/textarea/select
  const t = e.target;
  if (t.matches("input, textarea, select") && t.id !== "role-name-input") return;
  if (e.key === "?") { showHelp(); return; }
  if (e.key === "Escape") {
    if (document.getElementById("help-overlay").classList.contains("open")) hideHelp();
    if (document.getElementById("compare-overlay").classList.contains("open")) hideCompareModal();
    if (!document.getElementById("role-popover").hidden) closeRolePopover();
    return;
  }
  if (!state.candidates.length) return;
  if (e.key === "j" || e.key === "k") { moveSelection(e.key === "j" ? 1 : -1); e.preventDefault(); return; }
  if (state.selectedCandidateId != null && STAGE_KEYS[e.key]) {
    updateFunnelStage(state.selectedCandidateId, STAGE_KEYS[e.key]);
    e.preventDefault();
  }
  if (state.selectedCandidateId != null && (e.key === "0" || e.key === "Backspace")) {
    updateFunnelStage(state.selectedCandidateId, "");
    e.preventDefault();
  }
}

function moveSelection(delta) {
  if (!state.candidates.length) return;
  const visible = state.candidates.filter(c => {
    if (state.filter === "in_funnel") return state.funnel[String(c.id)];
    if (state.filter === "Hired") return state.funnel[String(c.id)] === "Hired";
    if (state.filter === "Rejected") return state.funnel[String(c.id)] === "Rejected";
    return true;
  });
  if (!visible.length) return;
  const curIdx = visible.findIndex(c => c.id === state.selectedCandidateId);
  let next = curIdx + delta;
  if (next < 0) next = 0;
  if (next >= visible.length) next = visible.length - 1;
  selectCandidate(visible[next].id);
  // Scroll the row into view
  const row = document.querySelector(`.candidate-row[data-id="${visible[next].id}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
}

// ─── Toast ─────────────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3500);
}

// ─── Render orchestrator ───────────────────────────────
function renderAll() {
  renderRoleSwitcher();
  renderJdPane();
  renderFunnelSummary();
  renderCandidates();
  renderDetail();
  if (state.drawerTab === "coverage") renderSkillCoverage();
  else renderFunnelBoard();
}

// Pull a clean "5+ yrs" / "Senior" label out of the extracted_experience array.
// Prefer numeric year mentions; fall back to a seniority token.
function formatExperience(exp) {
  if (!Array.isArray(exp) || !exp.length) return "";
  for (const e of exp) {
    const m = String(e).match(/(\d+)(\+?)\s*years?/i);
    if (m) return `${m[1]}${m[2]} yrs`;
  }
  const seniority = exp.find(e => /\b(senior|junior|principal|lead|mid[- ]?level|entry[- ]?level)\b/i.test(e));
  if (seniority) {
    const tok = seniority.match(/\b(senior|junior|principal|lead|mid[- ]?level|entry[- ]?level)\b/i)[0];
    return tok.charAt(0).toUpperCase() + tok.slice(1).toLowerCase();
  }
  return "";
}

// ─── Utils ─────────────────────────────────────────────
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
