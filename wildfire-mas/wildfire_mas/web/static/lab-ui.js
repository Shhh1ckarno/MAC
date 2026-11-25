"use strict";

const LAB_NOTES = {
  1: "Карта, агенты и каналы связи.",
  2: "Назначения штаба и сравнение двух механизмов на правилах на одном пожаре.",
  3: "8 расчетов, 1 место заправки, баки по 8 единиц. Сценарий массовых возгораний.",
  4: "Выберите сбой и режим. Недоступность LLM воспроизводится тестовым клиентом.",
};
const COMPARISON_COLORS = ["#3678bf", "#c36a25"];
const formatNumber = (value) => value == null ? "—" : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(value);
let lab = 1, comparisonRequest = null, activity = [], recovery = null;

function selectLab(number) {
  lab = number;
  for (const button of document.querySelectorAll("[data-lab-button]")) {
    button.setAttribute("aria-pressed", Number(button.dataset.labButton) === lab ? "true" : "false");
  }
  for (const element of document.querySelectorAll("[data-labs]")) {
    element.hidden = !element.dataset.labs.split(" ").includes(String(lab));
  }
  $("labNote").textContent = LAB_NOTES[lab];
  reset();
}

function resetLabPanels() {
  comparisonRequest?.abort();
  comparisonRequest = null;
  $("comparisonOutput").hidden = true;
  $("comparisonStatus").textContent = "";
  $("btnCompare").disabled = false;
  $("queue").replaceChildren();
  $("labActivity").replaceChildren();
  $("faultState").textContent = "";
  $("outcome").hidden = true;
  $("llmPanel").hidden = true;
  activity = [];
  recovery = null;
}

function table(headers, rows) {
  const result = document.createElement("table");
  const head = result.createTHead().insertRow();
  for (const title of headers) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = title;
    head.append(th);
  }
  const body = result.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) tr.insertCell().textContent = value;
  }
  return result;
}

function eventDescription(event) {
  switch (event.event) {
    case "refill_started":
      return `${event.crew} начал заправку на ${event.agent}. Ожидание: ${event.wait} такт. Порядок: ${{
        fifo: "по заявкам", priority: "по приоритету", aged: "после долгого ожидания",
      }[event.reason]}.`;
    case "refill_done": return `${event.crew} завершил заправку на ${event.agent}.`;
    case "refill_cancelled": return `Бронь ${event.crew} на ${event.agent} отменена.`;
    case "fault": return event.post ? `Отказ ${event.post} при заправке ${event.crew}.`
      : event.message != null ? `Штабу передано некорректное донесение № ${event.message}.` : "Тестовый сервис LLM недоступен.";
    case "report_rejected": return `Штаб отклонил донесение № ${event.message}.`;
    case "violation": return `Бронь осталась на недоступной точке ${event.post}.`;
    case "post_unavailable": return `${event.agent} исключил ${event.post} при выборе заправки.`;
    case "llm_error": return "Сервис LLM вернул ошибку.";
    case "llm_fallback": return "Штаб назначает расчеты по правилам.";
    default: return null;
  }
}

function updateLabPanels(tick) {
  if (lab !== 3 && lab !== 4) return;
  const posts = tick.water || [];
  $("queuePanel").hidden = lab === 4 && tick.fault?.case !== "water_outage";
  $("queue").replaceChildren(...posts.map((post) => {
    const block = document.createElement("section");
    const title = document.createElement("h4");
    title.textContent = post.broken ? `${post.id} · отказ`
      : `${post.id} · занято ${post.serving.length} из ${post.capacity} · заправлено ${post.served}`;
    const serving = document.createElement("p");
    serving.className = "muted small";
    serving.textContent = post.serving.length
      ? post.serving.map(({ crew, until }) => `${crew} до такта ${until}`).join(", ")
      : "Сейчас никто не заправляется";
    block.append(title, serving);
    if (post.requests.length) {
      block.append(table(["Расчет", "Приоритет", "Ожидание, такт."], post.requests.map((r) => [
        r.crew, formatNumber(r.priority), r.waiting ?? "в пути",
      ])));
    }
    return block;
  }));
  for (const event of tick.events || []) {
    const text = eventDescription(event);
    if (text) activity.push({ t: event.t, text });
  }
  if (tick.fault) {
    const fault = tick.fault;
    if (fault.recovery != null && fault.recovery !== recovery) {
      activity.push({ t: fault.recovery, text: "Восстановление зафиксировано." });
      recovery = fault.recovery;
    }
    const status = !fault.enabled ? "Штатный режим"
      : fault.start == null ? "Сбой еще не сработал"
      : fault.recovery == null ? `Сбой на такте ${fault.start}. Восстановление пока не зафиксировано.`
      : `Сбой на такте ${fault.start}. Восстановление на такте ${fault.recovery}.`;
    $("faultState").textContent = `${status} Нарушений: ${fault.violations}.`;
  }
  activity = activity.slice(-32);
  $("labActivity").replaceChildren(...activity.slice().reverse().map(({ t, text }) => {
    const row = document.createElement("li");
    row.textContent = `t=${t}  ${text}`;
    return row;
  }));
}

function drawComparison(data) {
  const canvas = $("comparisonChart");
  const g = canvas.getContext("2d");
  const style = getComputedStyle(canvas);
  const width = canvas.width, height = canvas.height;
  const left = 55, right = width - 18, top = 17, bottom = height - 37;
  const values = data.variants.flatMap((variant) => variant.series.map((point) => point[1]));
  const max = Math.max(1, ...values);
  const step = 10 ** Math.floor(Math.log10(max / 4));
  const upper = Math.ceil(max / step) * step;
  const x = (t) => left + t / data.horizon * (right - left);
  const y = (value) => bottom - value / upper * (bottom - top);
  g.clearRect(0, 0, width, height);
  g.font = "12px system-ui";
  g.fillStyle = style.color;
  g.strokeStyle = style.borderColor;
  for (let i = 0; i <= 4; i++) {
    const value = upper * i / 4;
    g.textAlign = "right";
    g.fillText(formatNumber(value), left - 8, y(value) + 4);
    g.beginPath(); g.moveTo(left, y(value)); g.lineTo(right, y(value)); g.stroke();
    g.textAlign = "center";
    const t = Math.round(data.horizon * i / 4);
    g.fillText(t, x(t), bottom + 16);
  }
  g.fillText("Такт", (left + right) / 2, height - 5);
  data.variants.forEach((variant, index) => {
    g.strokeStyle = g.fillStyle = COMPARISON_COLORS[index];
    g.lineWidth = 2.5;
    g.setLineDash(index === 0 ? [6, 4] : []);
    g.beginPath();
    variant.series.forEach(([t, value], i) => {
      if (i === 0) g.moveTo(x(t), y(value));
      else g.lineTo(x(t), y(value));
    });
    g.stroke();
    const last = variant.series.at(-1);
    if (last) {
      g.beginPath();
      g.arc(x(last[0]), y(last[1]), 3.3, 0, 2 * Math.PI);
      g.fill();
    }
  });
  g.setLineDash([]);
}

async function compareRuns() {
  const request = new AbortController();
  comparisonRequest?.abort();
  comparisonRequest = request;
  const selected = lab;
  $("comparisonOutput").hidden = true;
  $("comparisonStatus").textContent = "Выполняются два прогона…";
  $("btnCompare").disabled = true;
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: request.signal,
      body: JSON.stringify({ lab: selected, config: readConfig(), alternative: $("alternative").value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось выполнить сравнение");
    if (comparisonRequest !== request) return;
    $("comparisonStatus").textContent = `Seed ${data.seed} · горизонт ${data.horizon} тактов`;
    const legend = data.variants.map((variant, index) => {
      const item = document.createElement("span");
      const sample = document.createElement("i");
      sample.style.borderColor = COMPARISON_COLORS[index];
      if (index === 0) sample.style.borderTopStyle = "dashed";
      item.append(sample, variant.label);
      return item;
    });
    $("comparisonLegend").replaceChildren(...legend);
    const rows = data.variants.map((variant) => {
      const m = variant.metrics;
      const extra = selected === 2 ? formatNumber(m.response_time_mean)
        : selected === 3 ? formatNumber(m.refill_wait_mean) : `${variant.fault.violations} / ${formatNumber(variant.fault.recovery)}`;
      return [variant.label, formatNumber(m.J_burn), formatNumber(m.J_set), formatNumber(m.M_sent), extra];
    });
    const headers = ["Вариант", "Ущерб", "Поселки", "Сообщения", {
      2: "Реакция, такт.", 3: "Ожидание, такт.", 4: "Нарушения / такт восстановления",
    }[selected]];
    $("comparisonTable").replaceChildren(table(headers, rows));
    $("comparisonOutcome").textContent = data.variants.filter((variant) => variant.failure)
      .map((variant) => `${variant.label}. ${variant.failure}`).join(" ");
    $("comparisonOutput").hidden = false;
    drawComparison(data);
  } catch (error) {
    if (comparisonRequest === request) $("comparisonStatus").textContent = error.message;
  } finally {
    if (comparisonRequest === request) $("btnCompare").disabled = false;
  }
}
