"""Generate a self-contained HTML timeline from trajectory data."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def generate_timeline(db_path: str, output_path: str) -> str:
    """Query trajectory DB and produce a static HTML dashboard."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # 1. Daily activity heatmap data (last 12 months)
    cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    daily = {}
    for row in db.execute(
        "SELECT DATE(timestamp) as day, COUNT(*) as n FROM events WHERE timestamp >= ? GROUP BY day ORDER BY day",
        (cutoff,),
    ):
        daily[row["day"]] = row["n"]

    # 2. Monthly per-project activity
    monthly_project = {}
    for row in db.execute(
        """SELECT strftime('%Y-%m', e.timestamp) as month, p.name, COUNT(*) as n
           FROM events e JOIN projects p ON e.project_id = p.id
           WHERE e.timestamp >= ?
           GROUP BY month, p.name
           ORDER BY month, n DESC""",
        (cutoff,),
    ):
        month = row["month"]
        if month not in monthly_project:
            monthly_project[month] = {}
        monthly_project[month][row["name"]] = row["n"]

    # 3. Top projects overall
    top_projects = []
    for row in db.execute(
        """SELECT p.name, COUNT(*) as n, MIN(e.timestamp) as first, MAX(e.timestamp) as last
           FROM events e JOIN projects p ON e.project_id = p.id
           GROUP BY p.name ORDER BY n DESC LIMIT 20"""
    ):
        top_projects.append({
            "name": row["name"],
            "count": row["n"],
            "first": row["first"][:10] if row["first"] else "",
            "last": row["last"][:10] if row["last"] else "",
        })

    # 4. Recent events
    recent = []
    for row in db.execute(
        """SELECT e.timestamp, e.event_type, e.title, p.name as project
           FROM events e JOIN projects p ON e.project_id = p.id
           ORDER BY e.timestamp DESC LIMIT 100"""
    ):
        recent.append({
            "ts": row["timestamp"][:19],
            "type": row["event_type"],
            "title": row["title"][:120],
            "project": row["project"],
        })

    # 5. Summary stats
    total_events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    total_projects = db.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    date_range = db.execute("SELECT MIN(timestamp), MAX(timestamp) FROM events").fetchone()

    db.close()

    # Build months list for chart
    months_sorted = sorted(monthly_project.keys())
    # Get all project names that appear
    all_proj_names = set()
    for m in monthly_project.values():
        all_proj_names.update(m.keys())
    # Pick top 15 by total count for the chart
    proj_totals = {}
    for m in monthly_project.values():
        for p, n in m.items():
            proj_totals[p] = proj_totals.get(p, 0) + n
    top_chart_projects = sorted(proj_totals, key=proj_totals.get, reverse=True)[:15]

    # Monthly series for stacked bar
    monthly_series = {}
    for proj in top_chart_projects:
        monthly_series[proj] = [monthly_project.get(m, {}).get(proj, 0) for m in months_sorted]

    html = _render_html(
        daily=daily,
        months=months_sorted,
        monthly_series=monthly_series,
        top_projects=top_projects,
        recent=recent,
        total_events=total_events,
        total_projects=total_projects,
        date_range=(date_range[0][:10] if date_range[0] else "?", date_range[1][:10] if date_range[1] else "?"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    Path(output_path).write_text(html)
    return output_path


def _render_html(
    *,
    daily: dict,
    months: list,
    monthly_series: dict,
    top_projects: list,
    recent: list,
    total_events: int,
    total_projects: int,
    date_range: tuple,
    generated_at: str,
) -> str:
    # Color palette for projects
    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#d37295",
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trajectory Timeline</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 24px; font-size: 14px; }}
  .stats {{ display: flex; gap: 24px; margin-bottom: 24px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px 20px; }}
  .stat-value {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
  .stat-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ font-size: 16px; font-weight: 600; color: #c9d1d9; margin-bottom: 12px;
                    border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
  .heatmap {{ display: flex; flex-wrap: wrap; gap: 2px; }}
  .heatmap-day {{ width: 12px; height: 12px; border-radius: 2px; }}
  .heatmap-label {{ font-size: 10px; color: #8b949e; width: 100%; margin-top: 4px; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; }}
  canvas {{ max-width: 100%; }}
  .project-list {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 8px; }}
  .project-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px;
                   display: flex; justify-content: space-between; align-items: center; }}
  .project-name {{ font-weight: 600; font-size: 14px; }}
  .project-count {{ color: #58a6ff; font-weight: 700; }}
  .project-dates {{ font-size: 11px; color: #8b949e; }}
  .event-list {{ max-height: 500px; overflow-y: auto; }}
  .event-item {{ padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 13px; display: flex; gap: 12px; }}
  .event-ts {{ color: #8b949e; white-space: nowrap; min-width: 130px; }}
  .event-type {{ border-radius: 3px; padding: 1px 6px; font-size: 11px; font-weight: 600; }}
  .event-type-commit {{ background: #1f6feb33; color: #58a6ff; }}
  .event-type-conversation {{ background: #238636aa; color: #3fb950; }}
  .event-type-doc_change {{ background: #f0883e33; color: #f0883e; }}
  .event-type-archive {{ background: #8b949e33; color: #8b949e; }}
  .event-project {{ color: #bc8cff; min-width: 120px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .event-title {{ color: #c9d1d9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
</head>
<body>

<h1>Trajectory</h1>
<p class="subtitle">Generated {generated_at} &middot; {date_range[0]} to {date_range[1]}</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{total_events:,}</div><div class="stat-label">Total Events</div></div>
  <div class="stat"><div class="stat-value">{total_projects}</div><div class="stat-label">Projects</div></div>
  <div class="stat"><div class="stat-value">{len(daily)}</div><div class="stat-label">Active Days (12mo)</div></div>
</div>

<div class="section">
  <div class="section-title">Activity (last 12 months)</div>
  <div class="chart-container">
    <canvas id="heatmapChart" height="120"></canvas>
  </div>
</div>

<div class="section">
  <div class="section-title">Monthly Activity by Project</div>
  <div class="chart-container">
    <canvas id="monthlyChart" height="300"></canvas>
  </div>
</div>

<div class="section">
  <div class="section-title">Top Projects</div>
  <div class="project-list">
    {"".join(f'''<div class="project-item">
      <div><div class="project-name">{p["name"]}</div><div class="project-dates">{p["first"]} — {p["last"]}</div></div>
      <div class="project-count">{p["count"]:,}</div>
    </div>''' for p in top_projects)}
  </div>
</div>

<div class="section">
  <div class="section-title">Recent Activity</div>
  <div class="event-list">
    {"".join(f'''<div class="event-item">
      <span class="event-ts">{e["ts"]}</span>
      <span class="event-type event-type-{e["type"]}">{e["type"]}</span>
      <span class="event-project">{e["project"]}</span>
      <span class="event-title">{e["title"]}</span>
    </div>''' for e in recent)}
  </div>
</div>

<script>
const dailyData = {json.dumps(daily)};
const months = {json.dumps(months)};
const series = {json.dumps(monthly_series)};
const colors = {json.dumps(colors)};

// Heatmap as bar chart (daily activity last 12 months)
(() => {{
  const today = new Date();
  const labels = [];
  const values = [];
  const bgColors = [];
  for (let i = 364; i >= 0; i--) {{
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    labels.push(key);
    const v = dailyData[key] || 0;
    values.push(v);
    if (v === 0) bgColors.push('#161b22');
    else if (v <= 3) bgColors.push('#0e4429');
    else if (v <= 10) bgColors.push('#006d32');
    else if (v <= 25) bgColors.push('#26a641');
    else bgColors.push('#39d353');
  }}
  new Chart(document.getElementById('heatmapChart'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ data: values, backgroundColor: bgColors, borderWidth: 0 }}] }},
    options: {{
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        title: (items) => items[0].label,
        label: (item) => item.raw + ' events'
      }} }} }},
      scales: {{
        x: {{ display: false }},
        y: {{ display: false }}
      }},
      maintainAspectRatio: false,
      responsive: true,
      barPercentage: 1.0,
      categoryPercentage: 1.0,
    }}
  }});
}})();

// Monthly stacked bar
(() => {{
  const datasets = Object.entries(series).map(([name, data], i) => ({{
    label: name,
    data: data,
    backgroundColor: colors[i % colors.length],
  }}));
  new Chart(document.getElementById('monthlyChart'), {{
    type: 'bar',
    data: {{ labels: months, datasets }},
    options: {{
      plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
        y: {{ stacked: true, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
      }},
      maintainAspectRatio: false,
      responsive: true,
    }}
  }});
}})();
</script>

</body>
</html>"""
