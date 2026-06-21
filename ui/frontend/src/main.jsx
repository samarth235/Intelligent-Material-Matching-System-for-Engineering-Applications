import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowDownToLine,
  BarChart3,
  Boxes,
  BrainCircuit,
  Check,
  ClipboardList,
  Database,
  GitCompare,
  Layers3,
  Loader2,
  Search,
  SlidersHorizontal,
  Sparkles,
  Target,
} from "lucide-react";
import "./styles.css";

const API = "";

const properties = [
  ["Density_g_cm3", "Density", "g/cm3"],
  ["Yield_Strength_MPa", "Yield", "MPa"],
  ["Tensile_Strength_MPa", "Tensile", "MPa"],
  ["Hardness_HB", "Hardness", "HB"],
  ["Corrosion_Resistance_0_10", "Corrosion", "/10"],
  ["Thermal_Conductivity_W_mK", "Thermal", "W/mK"],
  ["Cost_Index_1_10", "Cost", "/10"],
  ["Machinability", "Machinability", "/10"],
];

const navItems = [
  { key: "recommend", label: "Recommend", icon: Sparkles },
  { key: "search", label: "Search", icon: Search },
  { key: "database", label: "Database", icon: Database },
  { key: "compare", label: "Compare", icon: GitCompare },
];

const initialWeights = {
  strength: 7,
  weight: 7,
  corrosion: 5,
  cost: 3,
  thermal: 3,
};

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function App() {
  const [activeView, setActiveView] = useState("recommend");
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/api/meta")
      .then(setMeta)
      .catch(() => setError("Unable to connect to the material API."));
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <strong>MatSelect</strong>
            <span>Material intelligence</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.key}
                type="button"
                className={activeView === item.key ? "nav-item active" : "nav-item"}
                onClick={() => setActiveView(item.key)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-panel">
          <span className="panel-label">Dataset</span>
          <strong>{meta?.stats?.dataset || "Loading"}</strong>
          <p>{meta ? `${meta.stats.totalMaterials} materials across ${meta.stats.applications} applications.` : "Preparing workspace."}</p>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Engineering decision support</p>
            <h1>Material selection workspace</h1>
          </div>
          <div className="topbar-stats">
            <Metric label="Materials" value={meta?.stats?.totalMaterials || "-"} />
            <Metric label="Alloy types" value={meta?.stats?.alloyTypes || "-"} />
            <Metric label="Applications" value={meta?.stats?.applications || "-"} />
          </div>
        </header>

        {error && <div className="notice error">{error}</div>}
        {!meta && !error && <LoadingState label="Loading material database" />}
        {meta && (
          <>
            {activeView === "recommend" && <RecommendView meta={meta} />}
            {activeView === "search" && <SearchView meta={meta} />}
            {activeView === "database" && <DatabaseView meta={meta} />}
            {activeView === "compare" && <CompareView meta={meta} />}
          </>
        )}
      </main>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function LoadingState({ label }) {
  return (
    <div className="empty-state">
      <Loader2 className="spin" size={26} />
      <p>{label}</p>
    </div>
  );
}

function RecommendView({ meta }) {
  const [weights, setWeights] = useState(initialWeights);
  const [application, setApplication] = useState("All");
  const [recommendations, setRecommendations] = useState([]);
  const [mlInfo, setMlInfo] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runRecommendation = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api("/api/recommend", {
        method: "POST",
        body: JSON.stringify({ weights, application, topN: 5 }),
      });
      const nextRecommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
      const hasPredictions = nextRecommendations.every(
        (item) => item.ML_Predicted_Application && typeof item.ML_Confidence === "number"
      );
      setRecommendations(nextRecommendations);
      setMlInfo({
        ...(data.ml || {}),
        active: Boolean(data.ml?.active && hasPredictions),
        errors:
          data.ml?.active && !hasPredictions
            ? ["The API response did not include ML predictions. Restart the backend on port 5050."]
            : data.ml?.errors || [],
      });
      setSelected(nextRecommendations[0]?.Material || null);
    } catch {
      setError("Unable to generate recommendations. Confirm that the backend is running on port 5050.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runRecommendation();
  }, []);

  const selectedMaterial = recommendations.find((item) => item.Material === selected) || recommendations[0];

  return (
    <section className="view-grid">
      <div className="control-panel">
        <SectionTitle icon={SlidersHorizontal} title="Recommendation controls" />
        <div className="field">
          <label htmlFor="application">Application</label>
          <select id="application" value={application} onChange={(event) => setApplication(event.target.value)}>
            <option>All</option>
            {meta.applications.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <WeightSlider label="Strength" value={weights.strength} onChange={(value) => setWeights({ ...weights, strength: value })} />
        <WeightSlider label="Low weight" value={weights.weight} onChange={(value) => setWeights({ ...weights, weight: value })} />
        <WeightSlider label="Corrosion resistance" value={weights.corrosion} onChange={(value) => setWeights({ ...weights, corrosion: value })} />
        <WeightSlider label="Low cost" value={weights.cost} onChange={(value) => setWeights({ ...weights, cost: value })} />
        <WeightSlider label="Thermal conductivity" value={weights.thermal} onChange={(value) => setWeights({ ...weights, thermal: value })} />
        <button type="button" className="primary-button" onClick={runRecommendation} disabled={loading}>
          {loading ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
          Generate recommendations
        </button>
      </div>

      <div className="content-panel">
        <SectionTitle icon={BarChart3} title="Top recommendations" />
        <MlBanner info={mlInfo} application={application} />
        {error && <div className="notice error">{error}</div>}
        {loading && <LoadingState label="Evaluating materials" />}
        {!loading && recommendations.length === 0 && <EmptyCopy title="No materials found" text="Adjust the application filter or priority weights." />}
        {!loading && recommendations.length > 0 && (
          <>
            <div className="recommendation-grid">
              {recommendations.slice(0, 3).map((item, index) => (
                <RecommendationCard key={item.Material} item={item} rank={index + 1} active={selected === item.Material} onClick={() => setSelected(item.Material)} />
              ))}
            </div>
            <div className="detail-layout">
              <RadarChart materials={recommendations.slice(0, 3)} />
              {selectedMaterial && <MaterialDetail item={selectedMaterial} />}
            </div>
            <DataTable
              rows={recommendations.slice(0, 5)}
              columns={properties.map(([key, label, unit]) => ({ key, label, unit }))}
              leadingColumns={["Material", "Alloy_Type", "ML_Predicted_Application", "ML_Confidence"]}
            />
          </>
        )}
      </div>
    </section>
  );
}

function MlBanner({ info, application }) {
  if (!info) return null;

  return (
    <div className={info.active ? "ml-banner" : "ml-banner muted"}>
      <BrainCircuit size={18} />
      <div>
        <strong>{info.active ? "ML model active" : "ML model unavailable"}</strong>
        <span>
          {info.active && application !== "All"
            ? `Ranking uses ${info.model} target probability for ${application}.`
            : info.active
              ? `Recommendations include ${info.model} predictions and confidence.`
              : info.errors?.[0] || "Using weighted database scoring only."}
        </span>
      </div>
    </div>
  );
}

function WeightSlider({ label, value, onChange }) {
  return (
    <div className="range-field">
      <div>
        <label>{label}</label>
        <span>{value}</span>
      </div>
      <input type="range" min="1" max="10" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

function RecommendationCard({ item, rank, active, onClick }) {
  return (
    <button type="button" className={active ? "result-card active" : "result-card"} onClick={onClick}>
      <div className="result-head">
        <span className="rank">#{rank}</span>
        <span className="score">{formatNumber(item.ML_Recommendation_Score || item.Recommendation_Score, 1)}</span>
      </div>
      <strong>{item.Material}</strong>
      <p>{item.Alloy_Type}</p>
      <div className="ml-card-row">
        <BrainCircuit size={14} />
        <span>{item.ML_Predicted_Application || "No prediction"}</span>
        {item.ML_Confidence !== null && item.ML_Confidence !== undefined && <b>{formatNumber(item.ML_Confidence * 100, 0)}%</b>}
      </div>
      <div className="reason-list">
        {(item.reasons || []).slice(0, 3).map((reason) => (
          <span key={reason}>
            <Check size={13} />
            {reason}
          </span>
        ))}
      </div>
    </button>
  );
}

function MaterialDetail({ item }) {
  const rows = [
    ["Density", `${formatNumber(item.Density_g_cm3, 2)} g/cm3`],
    ["Tensile strength", `${formatNumber(item.Tensile_Strength_MPa, 0)} MPa`],
    ["Yield strength", `${formatNumber(item.Yield_Strength_MPa, 0)} MPa`],
    ["Corrosion", `${formatNumber(item.Corrosion_Resistance_0_10, 1)}/10`],
    ["Cost index", `${formatNumber(item.Cost_Index_1_10, 1)}/10`],
    ["Machinability", `${formatNumber(item.Machinability, 1)}/10`],
  ];

  return (
    <article className="detail-card">
      <div className="detail-title">
        <Layers3 size={18} />
        <div>
          <strong>{item.Material}</strong>
          <span>{item.Primary_Application}</span>
        </div>
      </div>
      <div className="ml-detail">
        <Target size={16} />
        <div>
          <span>ML prediction</span>
          <strong>
            {item.ML_Predicted_Application || "Unavailable"}
            {item.ML_Confidence !== null && item.ML_Confidence !== undefined ? ` (${formatNumber(item.ML_Confidence * 100, 0)}%)` : ""}
          </strong>
        </div>
      </div>
      <dl className="property-list">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

function SearchView({ meta }) {
  const keys = Object.keys(meta.searchColumns);
  const [property, setProperty] = useState(keys[0]);
  const active = meta.searchColumns[property];
  const [min, setMin] = useState(active.min);
  const [max, setMax] = useState(active.max);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const next = meta.searchColumns[property];
    setMin(next.min);
    setMax(next.max);
  }, [property, meta.searchColumns]);

  const runSearch = async () => {
    setLoading(true);
    try {
      const data = await api("/api/search", {
        method: "POST",
        body: JSON.stringify({ property, min, max }),
      });
      setRows(data.materials);
      setTotal(data.total);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSearch();
  }, []);

  return (
    <section className="single-panel">
      <SectionTitle icon={Search} title="Advanced material search" />
      <div className="filter-row">
        <div className="field">
          <label>Property</label>
          <select value={property} onChange={(event) => setProperty(event.target.value)}>
            {keys.map((key) => (
              <option key={key} value={key}>
                {meta.searchColumns[key].label}
              </option>
            ))}
          </select>
        </div>
        <NumberField label="Minimum" value={min} onChange={setMin} />
        <NumberField label="Maximum" value={max} onChange={setMax} />
        <button type="button" className="primary-button compact" onClick={runSearch} disabled={loading}>
          {loading ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
          Search
        </button>
      </div>
      <div className="subtle-summary">{total} materials match the selected range.</div>
      <DataTable rows={rows} leadingColumns={["Material", "Alloy_Type", active.column, "Primary_Application"]} />
    </section>
  );
}

function NumberField({ label, value, onChange }) {
  return (
    <div className="field">
      <label>{label}</label>
      <input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </div>
  );
}

function DatabaseView({ meta }) {
  const [search, setSearch] = useState("");
  const [application, setApplication] = useState("All");
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ search, application, limit: "1000" });
    fetch(`/api/materials?${params.toString()}`, { signal: controller.signal })
      .then((response) => response.json())
      .then((data) => {
        setRows(data.materials);
        setTotal(data.total);
      })
      .catch((err) => {
        if (err.name !== "AbortError") console.error(err);
      });
    return () => controller.abort();
  }, [search, application]);

  const downloadCsv = () => {
    const headers = Object.keys(rows[0] || {});
    const csvRows = [
      headers.join(","),
      ...rows.map((row) => headers.map((key) => `"${String(row[key] ?? "").replaceAll('"', '""')}"`).join(",")),
    ];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `materials_${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="single-panel">
      <SectionTitle icon={Database} title="Material database" />
      <div className="filter-row">
        <div className="field grow">
          <label>Search by material name</label>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Aluminum, steel, titanium" />
        </div>
        <div className="field">
          <label>Application</label>
          <select value={application} onChange={(event) => setApplication(event.target.value)}>
            <option>All</option>
            {meta.applications.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <button type="button" className="secondary-button compact" onClick={downloadCsv} disabled={rows.length === 0}>
          <ArrowDownToLine size={17} />
          Export CSV
        </button>
      </div>
      <div className="subtle-summary">{total} materials shown.</div>
      <DataTable rows={rows} leadingColumns={["Material", "Alloy_Type", "Density_g_cm3", "Tensile_Strength_MPa", "Corrosion_Resistance_0_10", "Cost_Index_1_10", "Primary_Application"]} />
    </section>
  );
}

function CompareView({ meta }) {
  const initial = meta.materials.slice(0, 3);
  const [selected, setSelected] = useState([initial[0], initial[1], initial[2]].filter(Boolean));
  const [rows, setRows] = useState([]);

  const updateSelection = (index, value) => {
    const next = [...selected];
    next[index] = value;
    setSelected(Array.from(new Set(next.filter(Boolean))).slice(0, 3));
  };

  useEffect(() => {
    api("/api/compare", {
      method: "POST",
      body: JSON.stringify({ materials: selected }),
    }).then((data) => setRows(data.materials));
  }, [selected]);

  return (
    <section className="single-panel">
      <SectionTitle icon={GitCompare} title="Material comparison" />
      <div className="filter-row">
        {[0, 1, 2].map((index) => (
          <div className="field" key={index}>
            <label>Material {index + 1}</label>
            <select value={selected[index] || ""} onChange={(event) => updateSelection(index, event.target.value)}>
              <option value="">Not selected</option>
              {meta.materials.map((item) => (
                <option key={item} value={item} disabled={selected.includes(item) && selected[index] !== item}>
                  {item}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>
      <div className="detail-layout">
        <RadarChart materials={rows} />
        <ComparisonSummary rows={rows} />
      </div>
      <DataTable rows={rows} leadingColumns={["Material", "Alloy_Type", "Density_g_cm3", "Yield_Strength_MPa", "Tensile_Strength_MPa", "Hardness_HB", "Corrosion_Resistance_0_10", "Thermal_Conductivity_W_mK", "Cost_Index_1_10", "Machinability"]} />
    </section>
  );
}

function ComparisonSummary({ rows }) {
  if (!rows.length) return <EmptyCopy title="No comparison yet" text="Choose two or three distinct materials." />;
  return (
    <article className="detail-card">
      <div className="detail-title">
        <ClipboardList size={18} />
        <div>
          <strong>Selection notes</strong>
          <span>{rows.length} materials compared</span>
        </div>
      </div>
      <div className="note-list">
        {rows.map((row) => (
          <div key={row.Material}>
            <strong>{row.Material}</strong>
            <p>{(row.reasons || []).join(", ")}</p>
          </div>
        ))}
      </div>
    </article>
  );
}

function RadarChart({ materials }) {
  const labels = ["strength", "lowDensity", "corrosion", "lowCost", "thermal", "hardness"];
  const names = ["Strength", "Low density", "Corrosion", "Low cost", "Thermal", "Hardness"];
  const size = 320;
  const center = size / 2;
  const maxRadius = 116;
  const angleStep = (Math.PI * 2) / labels.length;
  const colors = ["#2563eb", "#0f766e", "#7c3aed"];

  const axisPoints = labels.map((_, index) => {
    const angle = -Math.PI / 2 + index * angleStep;
    return [center + Math.cos(angle) * maxRadius, center + Math.sin(angle) * maxRadius];
  });

  const rings = [0.25, 0.5, 0.75, 1].map((scale) =>
    axisPoints
      .map(([x, y]) => `${center + (x - center) * scale},${center + (y - center) * scale}`)
      .join(" ")
  );

  const polygons = materials.slice(0, 3).map((material, materialIndex) => {
    const points = labels
      .map((key, index) => {
        const angle = -Math.PI / 2 + index * angleStep;
        const value = Math.max(0, Math.min(10, Number(material.profile?.[key] ?? 0))) / 10;
        return `${center + Math.cos(angle) * maxRadius * value},${center + Math.sin(angle) * maxRadius * value}`;
      })
      .join(" ");
    return { points, color: colors[materialIndex], name: material.Material };
  });

  return (
    <article className="chart-card">
      <div className="chart-head">
        <strong>Property profile</strong>
        <span>Normalized 0-10 scale</span>
      </div>
      <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Normalized material property profile">
        {rings.map((points, index) => (
          <polygon key={index} points={points} className="radar-ring" />
        ))}
        {axisPoints.map(([x, y], index) => (
          <line key={names[index]} x1={center} y1={center} x2={x} y2={y} className="radar-axis" />
        ))}
        {polygons.map((polygon) => (
          <polygon key={polygon.name} points={polygon.points} fill={polygon.color} stroke={polygon.color} className="radar-shape" />
        ))}
        {axisPoints.map(([x, y], index) => (
          <text key={names[index]} x={center + (x - center) * 1.17} y={center + (y - center) * 1.17} textAnchor="middle" dominantBaseline="middle">
            {names[index]}
          </text>
        ))}
      </svg>
      <div className="legend">
        {polygons.map((polygon) => (
          <span key={polygon.name}>
            <i style={{ background: polygon.color }} />
            {polygon.name}
          </span>
        ))}
      </div>
    </article>
  );
}

function DataTable({ rows, leadingColumns, columns }) {
  const tableColumns = useMemo(() => {
    if (columns) return [...leadingColumns.map((key) => ({ key, label: readableLabel(key) })), ...columns];
    return leadingColumns.map((key) => ({ key, label: readableLabel(key) }));
  }, [leadingColumns, columns]);

  if (!rows.length) return <EmptyCopy title="No records" text="There are no materials for the current filters." />;

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {tableColumns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.Material}-${row.Alloy_Type}`}>
              {tableColumns.map((column) => (
                <td key={column.key}>
                  {column.key === "ML_Confidence" && typeof row[column.key] === "number"
                    ? `${formatNumber(row[column.key] * 100, 0)}%`
                    : typeof row[column.key] === "number"
                      ? formatNumber(row[column.key], column.unit ? 2 : 1)
                      : row[column.key] || "-"}
                  {column.unit && typeof row[column.key] === "number" ? <span className="unit"> {column.unit}</span> : null}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function readableLabel(key) {
  return key
    .replaceAll("_g_cm3", "")
    .replaceAll("_MPa", "")
    .replaceAll("_0_10", "")
    .replaceAll("_W_mK", "")
    .replaceAll("_1_10", "")
    .replaceAll("_HB", "")
    .replaceAll("_", " ");
}

function SectionTitle({ icon: Icon, title }) {
  return (
    <div className="section-title">
      <Icon size={19} />
      <h2>{title}</h2>
    </div>
  );
}

function EmptyCopy({ title, text }) {
  return (
    <div className="empty-state">
      <Boxes size={26} />
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
