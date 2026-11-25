"use strict";

const $ = (id) => document.getElementById(id);
const CELL_COLORS = ["#2f6b3a", "#a7c45f", "#b9a988", "#3d7fd6", "#d9b27c", "#5b6c8f"];
const PERF_COLORS = {
  INFORM: "#60a5fa", REQUEST: "#f59e0b", AGREE: "#34d399", REFUSE: "#f87171", FAILURE: "#ef4444",
  CFP: "#a78bfa", PROPOSE: "#e879f9", ACCEPT_PROPOSAL: "#22c55e", REJECT_PROPOSAL: "#fb7185", CANCEL: "#9ca3af",
};
const STATE_RU = {
  IDLE: "свободен", TO_FIRE: "к очагу", EXTINGUISH: "тушит", TO_WATER: "к воде", WAIT_SLOT: "ждёт слот",
  WAIT_WATER: "ждет доступную заправку", REFILL: "заправка", RETREAT: "отход", BROKEN: "неисправен", PATROL: "патруль", RETURN: "на базу",
  CHARGE: "зарядка", READY: "готова", BUSY: "занята", TO_BASE: "на базу", COORD: "ведет торги",
};

let ws = null, world = null, tick = null, cs = 18, llmConfigured = false;
const params = new URLSearchParams(location.search);
let autoplayUntil = params.has("autoplay") ? Number(params.get("autoplay")) : null;
const canvas = $("map"), ctx = canvas.getContext("2d");
const terrainLayer = document.createElement("canvas");

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("status").textContent = "соединение установлено"; reset(); };
  ws.onclose = () => { $("status").textContent = "соединение потеряно — перезагрузите страницу"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") {
      if (msg.lab !== lab) return;
      world = msg;
      tick = null;
      buildTerrain();
      $("summary").replaceChildren();
      if (autoplayUntil !== null) send({ cmd: "play", delay: 0.0 });
    }
    else if (msg.type === "tick") {
      if (!world || world.lab !== lab) return;
      tick = msg;
      render();
      updateLabPanels(tick);
      if (autoplayUntil !== null && tick.t >= autoplayUntil) { send({ cmd: "pause" }); autoplayUntil = null; }
    }
    else if (msg.type === "llm_request" && world?.lab === lab && world.config.hq_mode === "llm") {
      const model = msg.model || "LLM";
      $("llmNote").textContent = `${model}: ждем ответ с такта ${msg.t}. HTTP-попыток: ${msg.http_attempts}.`;
    }
    else if (msg.type === "summary" && world?.lab === lab) showSummary(msg.summary);
    else if ((msg.type === "outcome" && world?.lab === lab) || msg.type === "error") {
      $("outcome").textContent = msg.message;
      $("outcome").hidden = false;
    }
  };
}

const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
const delay = () => (100 - Number($("speed").value)) / 100 * 0.8 + 0.01;

function readConfig() {
  const num = (id) => Number($(id).value);
  const common = { seed: num("seed"), horizon: num("horizon") };
  if (lab === 3) return { ...common, refill_policy: $("refill_policy").value };
  if (lab === 4) return { ...common, fault_case: $("fault_case").value, fault_mode: $("fault_mode").value };
  return {
    ...common, scenario: $("scenario").value, mechanism: $("mechanism").value, hq_mode: $("hq_mode").value,
    n_crews: num("n_crews"), n_drones: num("n_drones"), wind_speed0: num("wind_speed0"), p_loss_cmd: num("p_loss_cmd"),
  };
}
function reset() {
  world = null;
  tick = null;
  resetLabPanels();
  showLLMConnection();
  send({ cmd: "reset", lab, config: readConfig() });
}

async function loadOptions() {
  const o = await (await fetch("/api/options")).json();
  for (const [id, choices] of Object.entries({
    scenario: o.scenarios, mechanism: o.mechanisms, alternative: o.mechanisms,
    refill_policy: o.refill_policies, fault_case: o.fault_cases, fault_mode: o.fault_modes,
  })) {
    $(id).replaceChildren(...choices.map(({ id, title }) => new Option(title, id)));
  }
  $("fault_mode").value = "after";
  for (const k of ["scenario", "mechanism", "seed", "hq_mode", "horizon", "n_crews", "n_drones",
    "refill_policy", "fault_case", "fault_mode"]) {
    if (params.has(k)) $(k).value = params.get(k);
  }
  $("alternative").value = $("mechanism").value === "cnp" ? "central_greedy" : "cnp";
  if (params.get("belief") === "1") $("showBelief").checked = true;
  llmConfigured = o.llm_configured;
}

function showLLMConnection() {
  $("llmNote").textContent = llmConfigured
    ? "Адрес LM Studio задан. Связь проверяется при первом запросе."
    : "Адрес LM Studio не настроен. Запустите сервер с LLM_BASE_URL и LLM_MODEL.";
}

function renderLLM(tick) {
  const result = tick.llm?.[0];
  const status = tick.llm_status;
  if (status) {
    const count = `HTTP-попыток: ${status.http_attempts}. Переходов на правила: ${status.fallbacks}.`;
    if (!status.configured) {
      $("llmNote").textContent = `Штаб использует правила, пока не задан LLM_BASE_URL. ${count}`;
    } else {
      const progress = !result ? "ждем открытый очаг и свободный расчет"
        : result.error ? `ошибка на такте ${result.t}`
        : `ответ на такте ${result.t}, принято назначений: ${result.valid?.length ?? 0}`;
      $("llmNote").textContent = `${status.model || "LLM"}: ${progress}. ${count}`;
    }
  }
  if (result) {
    $("llmPanel").hidden = false;
    $("llmPrompt").textContent = result.prompt || "";
    $("llmResponse").textContent = [
      result.response,
      result.error,
      result.violations?.length ? `Валидатор отклонил: ${result.violations.join(", ")}` : "",
    ].filter(Boolean).join("\n\n");
  }
}

function buildTerrain() {
  const W = world.width, H = world.height;
  cs = Math.max(8, Math.floor(Math.min(980 / W, 720 / H)));
  canvas.width = terrainLayer.width = W * cs;
  canvas.height = terrainLayer.height = H * cs;
  const g = terrainLayer.getContext("2d");
  for (let i = 0; i < W * H; i++) {
    const x = i % W, y = Math.floor(i / W), k = world.terrain[i];
    g.fillStyle = CELL_COLORS[k];
    g.fillRect(x * cs, y * cs, cs, cs);
    if (k === 0 && (x * 7 + y * 13) % 5 === 0) {
      g.fillStyle = "rgba(0,0,0,0.12)"; g.fillRect(x * cs + cs * 0.3, y * cs + cs * 0.3, cs * 0.35, cs * 0.35);
    }
    if (k === 4) {
      g.fillStyle = "#8a5a2b"; g.fillRect(x * cs + cs * 0.25, y * cs + cs * 0.35, cs * 0.5, cs * 0.45);
      g.beginPath(); g.moveTo(x * cs + cs * 0.15, y * cs + cs * 0.4); g.lineTo(x * cs + cs * 0.5, y * cs + cs * 0.1);
      g.lineTo(x * cs + cs * 0.85, y * cs + cs * 0.4); g.fill();
    }
  }
}

const cx = (p) => p[0] * cs + cs / 2, cy = (p) => p[1] * cs + cs / 2;

function fireColor(level) {
  const t = level / 9;
  const r = 255, gch = Math.round(190 - 150 * t), b = Math.round(40 - 30 * t);
  return `rgb(${r},${gch},${b})`;
}

function render() {
  if (!world || !tick) return;
  const W = world.width;
  ctx.drawImage(terrainLayer, 0, 0);
  ctx.fillStyle = "rgba(120,190,255,0.35)";
  for (const i of tick.wet) ctx.fillRect((i % W) * cs, Math.floor(i / W) * cs, cs, cs);
  for (const [i, ph, lv] of tick.cells) {
    const x = (i % W) * cs, y = Math.floor(i / W) * cs;
    if (ph === 1) { ctx.fillStyle = fireColor(lv); ctx.fillRect(x, y, cs, cs); }
    else if (ph === 2) { ctx.fillStyle = "#2b2b2b"; ctx.fillRect(x, y, cs, cs); }
    else if (ph === 3) { ctx.fillStyle = "#7b8fa1"; ctx.fillRect(x, y, cs, cs); }
  }
  if ($("showBelief").checked) {
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]);
    for (const i of tick.belief) ctx.strokeRect((i % W) * cs + 2, Math.floor(i / W) * cs + 2, cs - 4, cs - 4);
    ctx.setLineDash([]);
  }
  const agents = Object.fromEntries(tick.agents.map((a) => [a.id, a]));
  if ($("showZones").checked) {
    ctx.strokeStyle = "rgba(255,220,0,0.6)"; ctx.setLineDash([6, 4]);
    for (const a of tick.agents) if (a.kind === "drone" && a.zone) {
      const [x0, y0, x1, y1] = a.zone; ctx.strokeRect(x0 * cs + 1, y0 * cs + 1, (x1 - x0 + 1) * cs - 2, (y1 - y0 + 1) * cs - 2);
    }
    ctx.setLineDash([]);
  }
  if ($("showRadio").checked) {
    const R = world.config.radio_range;
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    for (const a of tick.agents) if (a.kind === "drone" || a.kind === "crew") {
      ctx.strokeRect((a.pos[0] - R) * cs, (a.pos[1] - R) * cs, (2 * R + 1) * cs, (2 * R + 1) * cs);
    }
  }
  if ($("showPaths").checked) {
    ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 1.5;
    for (const a of tick.agents) if (a.kind === "crew" && a.path && a.path.length) {
      ctx.beginPath(); ctx.moveTo(cx(a.pos), cy(a.pos));
      for (const p of a.path) ctx.lineTo(cx(p), cy(p));
      ctx.stroke();
    }
  }
  if ($("showMsgs").checked) {
    for (const m of tick.msgs) {
      const s = agents[m.from]; if (!s) continue;
      for (const r of m.to) {
        const d = agents[r]; if (!d) continue;
        ctx.strokeStyle = PERF_COLORS[m.perf] || "#fff"; ctx.globalAlpha = 0.75; ctx.lineWidth = 1.2;
        if (m.ch === "radio") ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(cx(s.pos), cy(s.pos)); ctx.lineTo(cx(d.pos), cy(d.pos)); ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
    }
  }
  ctx.font = `bold ${Math.max(9, cs * 0.55)}px system-ui`; ctx.textBaseline = "bottom";
  for (const a of tick.agents) {
    const x = cx(a.pos), y = cy(a.pos), r = cs * 0.42;
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#111";
    if (a.kind === "hq") { ctx.fillStyle = "#3d4c6b"; ctx.fillRect(x - r, y - r, 2 * r, 2 * r); ctx.strokeRect(x - r, y - r, 2 * r, 2 * r); }
    else if (a.kind === "water") {
      ctx.fillStyle = a.broken ? "#7f2635" : "#1f6fd6";
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
      ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
      if (a.broken) {
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r);
        ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r);
        ctx.stroke();
      }
    }
    else if (a.kind === "crew") {
      ctx.fillStyle = a.broken ? "#777" : "#d6302d"; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke();
      const frac = a.water / world.config.crew_tank;
      ctx.fillStyle = "#111"; ctx.fillRect(x - r, y + r + 1, 2 * r, 3);
      ctx.fillStyle = "#4fb3ff"; ctx.fillRect(x - r, y + r + 1, 2 * r * frac, 3);
    } else if (a.kind === "drone") {
      ctx.fillStyle = a.broken ? "#777" : "#ffd21f"; ctx.beginPath();
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r * 0.8); ctx.lineTo(x - r, y + r * 0.8); ctx.closePath(); ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
    ctx.strokeText(a.id, x + r * 0.6, y - r * 0.4); ctx.fillText(a.id, x + r * 0.6, y - r * 0.4);
  }
  const ang = tick.wind.deg * Math.PI / 180, wx = canvas.width - 40, wy = 40, L = 22;
  ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff"; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(wx, wy, 28, 0, 7); ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fill();
  ctx.beginPath(); ctx.moveTo(wx - L * Math.cos(ang), wy - L * Math.sin(ang)); ctx.lineTo(wx + L * Math.cos(ang), wy + L * Math.sin(ang)); ctx.stroke();
  ctx.beginPath(); ctx.fillStyle = "#fff";
  ctx.moveTo(wx + L * Math.cos(ang), wy + L * Math.sin(ang));
  ctx.lineTo(wx + (L - 9) * Math.cos(ang) - 6 * Math.sin(ang), wy + (L - 9) * Math.sin(ang) + 6 * Math.cos(ang));
  ctx.lineTo(wx + (L - 9) * Math.cos(ang) + 6 * Math.sin(ang), wy + (L - 9) * Math.sin(ang) - 6 * Math.cos(ang));
  ctx.fill();

  renderPanels(agents);
}

function renderPanels(agents) {
  $("clock").textContent = `t = ${tick.t} / ${world.config.horizon}`;
  $("wind").textContent = `ветер ${tick.wind.speed} м/с`;
  const m = tick.metrics || {};
  const rows = [
    ["Горит клеток", m.burning], ["Выгорело клеток", m.burnt_cells], ["Ущерб (ценность)", m.burnt_value],
    ["Потеряно посёлков", m.settlements_lost], ["Потушено клеток", m.extinguished_cells],
    ["Сообщений (отпр./дост.)", `${m.msgs_sent ?? 0} / ${m.msgs_delivered ?? 0}`],
  ];
  $("metrics").innerHTML = rows.map(([k, v]) => `<div>${k}</div><div>${v ?? "—"}</div>`).join("");

  const hq = agents["h"];
  $("tasks").innerHTML = hq && hq.tasks.length
    ? `<table><tr><th>Очаг</th><th>Клеток</th><th>Приоритет</th><th>Нужно</th><th>Назначены</th></tr>` +
      hq.tasks.map((t) => `<tr><td>${t.id}</td><td>${t.cells}</td><td>${t.priority}</td><td>${t.demand}</td>
        <td>${Object.entries(t.assigned).map(([c, s]) => s === "active" ? c : c + "?").join(", ") || "—"}</td></tr>`).join("") + `</table>`
    : `<span class="muted">активных очагов нет (по данным штаба)</span>`;

  $("agents").innerHTML = `<table><tr><th>Агент</th><th>Состояние</th><th>Ресурс</th><th>Задача</th></tr>` +
    tick.agents.filter((a) => a.kind !== "hq").map((a) => {
      let res = "";
      if (a.kind === "crew") res = `<span class="bar" style="width:${Math.round(40 * a.water / world.config.crew_tank)}px"></span> ${a.water}`;
      if (a.kind === "drone") res = `🔋 ${a.battery}`;
      if (a.kind === "water") res = `очередь ${a.queue.length}`;
      return `<tr><td>${a.id}</td><td>${a.broken ? "неисправен" : (STATE_RU[a.state] || a.state)}</td><td>${res}</td><td>${a.task || ""}</td></tr>`;
    }).join("") + `</table>`;

  $("msgs").innerHTML = tick.msgs.slice(-40).reverse().map((x) =>
    `<div><span class="muted">t=${x.t}</span> ${x.from} → ${x.to.join(",")} <b class="perf-${x.perf}">${x.perf}</b> ${x.type}</div>`).join("")
    || `<span class="muted">нет</span>`;
  $("log").replaceChildren(...tick.log.slice().reverse().map((text) => {
    const line = document.createElement("div");
    line.textContent = text;
    return line;
  }));

  renderLLM(tick);
}

function showSummary(s) {
  const rows = [
    ["Ущерб J_burn", s.J_burn], ["Без агентов (база)", s.J_burn_base], ["Спасено (Save)", s.Save],
    ["Потеряно посёлков", s.J_set], ["Очагов / ликвидировано", `${s.incidents} / ${s.incidents_extinguished}`],
    ["Ср. время ликвидации", s.T_ext_mean], ["Ср. задержка обнаружения", s.detect_latency_mean],
    ["Ср. время реагирования", s.response_time_mean], ["Сообщений доставлено", s.M_delivered],
    ["Время координации, мс", s.coord_time_ms], ["Индекс Джини загрузки", s.gini_busy],
    ["Ожидание на заправке (ср./макс.)", `${s.refill_wait_mean} / ${s.refill_wait_max}`],
  ];
  $("summary").innerHTML = `<h3 style="margin-top:10px">Итоги прогона</h3><table>` +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v ?? "—"}</td></tr>`).join("") + `</table>`;
}

$("btnReset").onclick = reset;
$("btnPlay").onclick = () => send({ cmd: "play", delay: delay() });
$("btnPause").onclick = () => send({ cmd: "pause" });
$("btnStep").onclick = () => send({ cmd: "step" });
$("speed").oninput = () => send({ cmd: "speed", delay: delay() });
for (const id of ["showBelief", "showMsgs", "showRadio", "showPaths", "showZones"]) $(id).onchange = render;

for (const button of document.querySelectorAll("[data-lab-button]")) {
  button.onclick = () => selectLab(Number(button.dataset.labButton));
}
for (const id of ["hq_mode", "refill_policy", "fault_case", "fault_mode"]) $(id).onchange = reset;
$("mechanism").onchange = () => {
  if ($("alternative").value === $("mechanism").value) {
    $("alternative").value = $("mechanism").value === "cnp" ? "central_greedy" : "cnp";
  }
};
$("btnCompare").onclick = compareRuns;
loadOptions().then(() => {
  const initial = Number(params.get("lab"));
  selectLab([1, 2, 3, 4].includes(initial) ? initial : 1);
  connect();
}).catch(() => { $("status").textContent = "не удалось загрузить настройки"; });
