const colorByType = {
  disease: "#bf3f3f",
  symptom: "#b7791f",
  drug: "#2563eb",
  test: "#7c3aed",
  treatment: "#15803d",
  department: "#0f766e",
  risk_factor: "#475467"
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function renderMetrics(stats, quality) {
  const items = [
    ["Entities", stats.entity_count],
    ["Relations", stats.relation_count],
    ["Records", stats.source_record_count],
    ["Entity Types", Object.keys(stats.entities_by_type || {}).length],
    ["Quality", quality?.passed ? "Pass" : "Review"]
  ];
  document.getElementById("metrics").innerHTML = items.map(([label, value]) => `
    <div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>
  `).join("");
}

function renderGraph(graph) {
  const svg = document.getElementById("graph");
  svg.innerHTML = "";
  const width = 920;
  const height = 520;
  const entities = graph.entities.slice(0, 54);
  const relations = graph.relations.filter(rel =>
    entities.some(entity => entity.id === rel.subject_id) &&
    entities.some(entity => entity.id === rel.object_id)
  ).slice(0, 95);
  const center = {x: width / 2, y: height / 2};
  const radius = 205;
  const positions = {};
  entities.forEach((entity, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(entities.length, 1);
    positions[entity.id] = {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius
    };
  });

  relations.forEach(rel => {
    const source = positions[rel.subject_id];
    const target = positions[rel.object_id];
    if (!source || !target) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", source.x);
    line.setAttribute("y1", source.y);
    line.setAttribute("x2", target.x);
    line.setAttribute("y2", target.y);
    line.setAttribute("class", "edge");
    svg.appendChild(line);
  });

  entities.forEach(entity => {
    const point = positions[entity.id];
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", point.x);
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", entity.type === "disease" ? 10 : 7);
    circle.setAttribute("fill", colorByType[entity.type] || "#667085");
    circle.setAttribute("class", "node");
    circle.innerHTML = `<title>${entity.label}: ${entity.name}</title>`;
    svg.appendChild(circle);

    if (entity.type !== "disease") return;
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", point.x + 12);
    label.setAttribute("y", point.y + 4);
    label.setAttribute("class", "node-label");
    label.textContent = entity.name;
    svg.appendChild(label);
  });
}

function renderChart(result) {
  const rows = result.rows || [];
  const max = Math.max(1, ...rows.map(row => Number(row.value || 0)));
  document.getElementById("chart").innerHTML = rows.map(row => {
    const value = Number(row.value || 0);
    const width = Math.max(4, Math.round((value / max) * 100));
    return `
      <div class="bar-row">
        <div class="bar-label">${row.name}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <div>${value}</div>
      </div>
    `;
  }).join("");
  document.getElementById("narrative").textContent = result.narrative || "";
}

function renderAnswer(result) {
  const evidence = (result.evidence || []).map(item =>
    `<li>${item.subject} - ${item.predicate} - ${item.object}<br><span>${item.evidence}</span></li>`
  ).join("");
  document.getElementById("answer").innerHTML = `
    <strong>${result.answer}</strong>
    <ol>${evidence}</ol>
  `;
}

async function loadGraph() {
  const [payload, quality] = await Promise.all([api("/api/graph"), api("/api/quality")]);
  renderMetrics(payload.stats, quality);
  renderGraph(payload.graph);
  document.getElementById("graphMeta").textContent = `${payload.stats.relation_count} relations`;
}

async function runPipeline() {
  document.getElementById("runDemo").disabled = true;
  await api("/api/pipelines/run", {
    method: "POST",
    body: JSON.stringify({task: "完成医疗数据处理、知识图谱生成问答和图谱驱动分析展示"})
  });
  await loadGraph();
  document.getElementById("runDemo").disabled = false;
}

async function analyze() {
  const question = document.getElementById("analysisQuestion").value;
  renderChart(await api("/api/analyze", {method: "POST", body: JSON.stringify({question})}));
}

async function ask() {
  const question = document.getElementById("qaQuestion").value;
  renderAnswer(await api("/api/qa", {method: "POST", body: JSON.stringify({question})}));
}

document.getElementById("runDemo").addEventListener("click", runPipeline);
document.getElementById("analyze").addEventListener("click", analyze);
document.getElementById("ask").addEventListener("click", ask);

loadGraph().then(() => Promise.all([analyze(), ask()])).catch(error => {
  document.getElementById("metrics").innerHTML = `<div class="metric"><div class="label">Error</div><div class="value">!</div></div>`;
  document.getElementById("narrative").textContent = error.message;
});
