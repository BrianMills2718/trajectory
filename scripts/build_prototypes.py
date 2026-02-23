"""Build 4 layout prototypes for the Expansions page."""
import json, sys
sys.path.insert(0, "/home/brian/projects/trajectory")
from pathlib import Path
from llm_client import call_llm, render_prompt
from trajectory.output.project_narrative import _strip_json_fences, PROMPTS_DIR

MODEL = "gemini/gemini-2.5-flash"

# ── Concept data ──
CONCEPTS = {
    # Core
    "reasoning_scarcity": {"layer":"core", "label":"Reasoning Scarcity", "desc":"Personal reasoning is the last scarce AI data. Compute scales, public data exhausted. How a specific person thinks has never been captured."},
    "memory_reconstruction": {"layer":"core", "label":"Memory Reconstruction", "desc":"Memory reconstructs, not records (Bartlett 1930s). Each recall rewrites. Real-time capture differs from recalled reasoning."},
    "observer_problem": {"layer":"core", "label":"Observer Problem", "desc":"2,400 years of failed self-observation. Socrates, Franklin, Freud — observer always compromised. AI breaks this."},
    "pharmakon": {"layer":"core", "label":"Pharmakon", "desc":"Every mind-extending tool reshapes the mind. Poison and remedy."},
    "data_moat": {"layer":"core", "label":"Data Moat", "desc":"AI capability commoditizes. Reasoning data does not."},
    "tacit_knowledge": {"layer":"core", "label":"Tacit Knowledge", "desc":"Klein RPD: experts pattern-match, can't self-report. Nonaka SECI: tacit transfer needs shared experience."},
    "structured_capture": {"layer":"core", "label":"Structured Capture", "desc":"Bridgewater Dot Collector. Tetlock superforecasters. Parrish Decision Journal. Structured methods produce different data."},
    "cognitive_infrastructure": {"layer":"core", "label":"Cognitive Infrastructure", "desc":"Each delegated layer without own data erodes supervision of the layer above."},
    "wellbeing_bridge": {"layer":"core", "label":"Wellbeing Bridge", "desc":"Woebot/Wysa evidence. Rogerian effect. Cognitive Mirror (2025). Capture and wellbeing converge."},
    # Connections
    "metacognitive_intervention": {"layer":"connection", "label":"Metacognitive Intervention", "desc":"observer_problem + wellbeing_bridge → Chen 2025: externalizing reasoning is itself a metacognitive intervention. Capture IS therapy.", "bridges":["observer_problem","wellbeing_bridge"]},
    "compound_returns": {"layer":"connection", "label":"Compound Returns", "desc":"data_moat + structured_capture → data compounds temporally (weeks→years) and structurally (cross-reference across people).", "bridges":["data_moat","structured_capture"]},
    "selection_transparency": {"layer":"connection", "label":"Selection Transparency", "desc":"tacit_knowledge + curator → captured data reflects comfort with externalizing. Most valuable reasoning is least likely shared.", "bridges":["tacit_knowledge","data_moat"]},
    "delegation_erosion": {"layer":"connection", "label":"Delegation Erosion", "desc":"cognitive_infrastructure + AI dependency → can't execute → can't strategize → can't judge. Specific causal chain.", "bridges":["cognitive_infrastructure","data_moat"]},
    # Frontier
    "counterfactual_capture": {"layer":"frontier", "label":"Counterfactual Capture", "desc":"Why not B, C, D encodes the risk model, constraints. The decision landscape, not the chosen path.", "extends":"structured_capture"},
    "pre_commitment": {"layer":"frontier", "label":"Pre-commitment", "desc":"State criteria before the emotional moment. Track follow-through. Behavioral economics for personal AI.", "extends":"structured_capture"},
    "meta_reasoning": {"layer":"frontier", "label":"Meta-reasoning", "desc":"What to reason about at all. Attention allocation. Highest-leverage cognitive function.", "extends":"cognitive_infrastructure"},
    "embodied_cognition": {"layer":"frontier", "label":"Embodied Cognition", "desc":"Damasio somatic markers. Biometric + cognitive data combined.", "extends":"tacit_knowledge"},
    "inter_brain_reasoning": {"layer":"frontier", "label":"Inter-brain Reasoning", "desc":"Board meetings where models collide. Individual reasoning is monologue. Frontier is dialectic.", "extends":"tacit_knowledge"},
    "generative_reasoning": {"layer":"frontier", "label":"Generative Reasoning", "desc":"Where insights come from. Framing shifts. Non-linear, unconscious.", "extends":"observer_problem"},
    "strategic_forgetting": {"layer":"frontier", "label":"Strategic Forgetting", "desc":"What to let decay. Curation includes release.", "extends":"data_moat"},
    "self_evolution": {"layer":"frontier", "label":"Self-evolution", "desc":"Model of 35-year-old reasoning informing 45-year-old. Growth, not crystallization.", "extends":"data_moat"},
    "reasoning_commons": {"layer":"frontier", "label":"Reasoning Commons", "desc":"Multiple people → cross-reference → 'how do the best differ from average?' Wedge → platform.", "extends":"compound_returns"},
    "reasoning_inheritance": {"layer":"frontier", "label":"Reasoning Inheritance", "desc":"Successors get reasoning, not playbook. Transferable cognitive model.", "extends":"data_moat"},
    "adversarial_awareness": {"layer":"frontier", "label":"Adversarial Awareness", "desc":"Legible cognition is a design surface. Who sees what, under what conditions.", "extends":"data_moat"},
}

# Build edges
EDGES = []
for cid, c in CONCEPTS.items():
    if "bridges" in c:
        for b in c["bridges"]:
            EDGES.append({"from": b, "to": cid, "label": "connects to", "style": "dashed"})
    if "extends" in c:
        EDGES.append({"from": c["extends"], "to": cid, "label": "extends into", "style": "solid"})

# Add some core→core edges
EDGES += [
    {"from":"reasoning_scarcity","to":"data_moat","label":"implies","style":"solid"},
    {"from":"memory_reconstruction","to":"observer_problem","label":"explains why","style":"solid"},
    {"from":"observer_problem","to":"structured_capture","label":"solved by","style":"solid"},
    {"from":"tacit_knowledge","to":"structured_capture","label":"requires","style":"solid"},
    {"from":"data_moat","to":"cognitive_infrastructure","label":"protects","style":"solid"},
]

NODES = [{"id":cid, "label":c["label"], "type":c["layer"]} for cid, c in CONCEPTS.items()]
GRAPH = {"nodes": NODES, "edges": EDGES}

# Colors (Wong palette)
COLORS = {"core":"#56B4E9", "connection":"#E69F00", "frontier":"#CC79A7"}

SHARED_CSS = """
:root {
    --bg: #0d1117; --bg-card: #161b22; --border: #21262d;
    --text: #c9d1d9; --text-bright: #e6edf3; --text-dim: #8b949e; --text-dimmer: #484f58;
    --core: #56B4E9; --connection: #E69F00; --frontier: #CC79A7;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; overflow-x:hidden; }
"""

DAGRE_CDN = '<script src="https://d3js.org/d3.v7.min.js"></script><script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>'

GRAPH_JS = f"""
const GRAPH = {json.dumps(GRAPH)};
const COLORS = {json.dumps(COLORS)};
const CONCEPTS = {json.dumps(CONCEPTS)};

function renderGraph(svgSelector, opts = {{}}) {{
    const svg = d3.select(svgSelector);
    const container = svg.node().parentElement;
    const W = opts.width || container.clientWidth;
    const H = opts.height || container.clientHeight;
    svg.attr('width', W).attr('height', H);

    const g = new dagre.graphlib.Graph();
    g.setGraph({{ rankdir: 'TB', nodesep: 60, ranksep: 80, marginx: 40, marginy: 40 }});
    g.setDefaultEdgeLabel(() => ({{}}));

    const NW = 140, NH = 40;
    const visibleLayers = opts.layers || ['core','connection','frontier'];
    const visibleNodes = GRAPH.nodes.filter(n => visibleLayers.includes(n.type));
    const visibleIds = new Set(visibleNodes.map(n => n.id));

    visibleNodes.forEach(n => g.setNode(n.id, {{ width: NW, height: NH }}));
    const visibleEdges = GRAPH.edges.filter(e => visibleIds.has(e.from) && visibleIds.has(e.to));
    visibleEdges.forEach(e => g.setEdge(e.from, e.to, {{ label: e.label }}));
    dagre.layout(g);

    // Center
    const bounds = {{ minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }};
    visibleNodes.forEach(n => {{
        const nd = g.node(n.id); if (!nd) return;
        bounds.minX = Math.min(bounds.minX, nd.x - NW/2);
        bounds.minY = Math.min(bounds.minY, nd.y - NH/2);
        bounds.maxX = Math.max(bounds.maxX, nd.x + NW/2);
        bounds.maxY = Math.max(bounds.maxY, nd.y + NH/2);
    }});
    const gW = bounds.maxX - bounds.minX + 80;
    const gH = bounds.maxY - bounds.minY + 80;
    const scale = Math.min(W / gW, H / gH, 1.2);
    const tx = (W - gW * scale) / 2 - bounds.minX * scale + 40 * scale;
    const ty = (H - gH * scale) / 2 - bounds.minY * scale + 40 * scale;

    svg.selectAll('*').remove();
    const root = svg.append('g').attr('transform', `translate(${{tx}},${{ty}}) scale(${{scale}})`);

    // Edges
    const edgeG = root.append('g');
    visibleEdges.forEach(e => {{
        const pts = g.edge(e.from, e.to);
        if (!pts || !pts.points) return;
        const line = d3.line().x(p=>p.x).y(p=>p.y).curve(d3.curveBasis);
        edgeG.append('path').attr('d', line(pts.points))
            .attr('fill','none').attr('stroke','#30363d').attr('stroke-width',1.5)
            .attr('stroke-dasharray', e.style==='dashed'?'6 3':'none')
            .attr('marker-end','url(#arrow)');
        // Edge label at midpoint
        const mid = pts.points[Math.floor(pts.points.length/2)];
        edgeG.append('text').attr('x',mid.x).attr('y',mid.y-6)
            .attr('text-anchor','middle').attr('fill','#484f58').attr('font-size','9px')
            .text(e.label || '');
    }});

    // Arrow marker
    svg.append('defs').append('marker').attr('id','arrow').attr('viewBox','0 0 10 10')
        .attr('refX',10).attr('refY',5).attr('markerWidth',6).attr('markerHeight',6)
        .attr('orient','auto').append('path').attr('d','M 0 0 L 10 5 L 0 10 z').attr('fill','#30363d');

    // Nodes
    const nodeG = root.append('g');
    visibleNodes.forEach(n => {{
        const nd = g.node(n.id); if (!nd) return;
        const color = COLORS[n.type];
        const ng = nodeG.append('g')
            .attr('transform', `translate(${{nd.x}},${{nd.y}})`)
            .attr('data-concept', n.id)
            .attr('class', 'graph-node')
            .style('cursor','pointer');

        if (n.type === 'connection') {{
            // Diamond for connections
            const hw=NW/2+10, hh=NH/2+6;
            ng.append('polygon')
                .attr('points', `0,${{-hh}} ${{hw}},0 0,${{hh}} ${{-hw}},0`)
                .attr('fill','var(--bg-card)').attr('stroke',color).attr('stroke-width',2);
        }} else {{
            ng.append('rect')
                .attr('x',-NW/2).attr('y',-NH/2).attr('width',NW).attr('height',NH)
                .attr('rx',6).attr('fill','var(--bg-card)').attr('stroke',color).attr('stroke-width', n.type==='frontier'?2.5:1.5);
            if (n.type === 'frontier') {{
                ng.select('rect').style('filter', `drop-shadow(0 0 6px ${{color}}40)`);
            }}
        }}

        // Label
        const words = n.label.split(' ');
        if (words.length <= 3) {{
            ng.append('text').attr('text-anchor','middle').attr('dy','0.35em')
                .attr('fill',color).attr('font-size','11px').attr('font-weight',500)
                .text(n.label);
        }} else {{
            const mid = Math.ceil(words.length/2);
            ng.append('text').attr('text-anchor','middle').attr('dy','-0.2em')
                .attr('fill',color).attr('font-size','11px').attr('font-weight',500)
                .text(words.slice(0,mid).join(' '));
            ng.append('text').attr('text-anchor','middle').attr('dy','1.0em')
                .attr('fill',color).attr('font-size','11px').attr('font-weight',500)
                .text(words.slice(mid).join(' '));
        }}

        // Hover
        ng.on('mouseenter', () => {{
            ng.select('rect,polygon').transition().duration(150)
                .attr('stroke-width',3).style('filter',`drop-shadow(0 0 12px ${{color}}80)`);
            document.querySelectorAll(`.concept-chip[data-concept="${{n.id}}"]`)
                .forEach(c => c.classList.add('hover-linked'));
            if (opts.onHover) opts.onHover(n.id);
        }});
        ng.on('mouseleave', () => {{
            ng.select('rect,polygon').transition().duration(150)
                .attr('stroke-width', n.type==='frontier'?2.5:1.5)
                .style('filter', n.type==='frontier'?`drop-shadow(0 0 6px ${{color}}40)`:'none');
            document.querySelectorAll('.hover-linked').forEach(c => c.classList.remove('hover-linked'));
            if (opts.onLeave) opts.onLeave(n.id);
        }});
        if (opts.onClick) ng.on('click', () => opts.onClick(n.id));
    }});

    return {{ g, scale, tx, ty }};
}}
"""

LEGEND_HTML = """
<div class="legend">
    <div class="legend-item"><span class="dot" style="background:var(--core)"></span> Your ideas</div>
    <div class="legend-item"><span class="dot" style="background:var(--connection)"></span> Connections drawn</div>
    <div class="legend-item"><span class="dot" style="background:var(--frontier)"></span> New territory</div>
</div>
"""

LEGEND_CSS = """
.legend { display:flex; gap:20px; font-size:12px; color:var(--text-dim); }
.legend-item { display:flex; align-items:center; gap:6px; }
.dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
"""

desk = Path("/mnt/c/Users/Brian/Desktop")

# ═══════════════════════════════════════════
# PROTOTYPE A: Graph-first with sidebar
# ═══════════════════════════════════════════
print("A: Graph + sidebar...")
a_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Expansions — Graph + Sidebar</title>
<style>{SHARED_CSS}
{LEGEND_CSS}
.layout {{ display:flex; height:100vh; }}
.graph-panel {{ flex:1; position:relative; }}
.graph-panel svg {{ width:100%; height:100%; }}
.sidebar {{ width:0; overflow:hidden; transition:width 0.3s; background:var(--bg-card); border-left:1px solid var(--border); }}
.sidebar.open {{ width:360px; }}
.sidebar-inner {{ width:360px; padding:32px 28px; overflow-y:auto; height:100%; }}
.sidebar-close {{ position:absolute; top:12px; right:12px; background:none; border:none; color:var(--text-dimmer); font-size:20px; cursor:pointer; }}
.sidebar-label {{ font-size:15px; font-weight:600; color:var(--text-bright); margin-bottom:8px; }}
.sidebar-layer {{ font-size:11px; text-transform:uppercase; letter-spacing:1px; padding:2px 8px; border-radius:4px; display:inline-block; margin-bottom:12px; }}
.sidebar-layer.core {{ background:rgba(86,180,233,0.15); color:var(--core); }}
.sidebar-layer.connection {{ background:rgba(230,159,0,0.15); color:var(--connection); }}
.sidebar-layer.frontier {{ background:rgba(204,121,167,0.15); color:var(--frontier); }}
.sidebar-desc {{ font-size:14px; line-height:1.7; color:var(--text); }}
.sidebar-connections {{ margin-top:16px; font-size:12px; color:var(--text-dim); }}
.sidebar-connections strong {{ color:var(--text-dimmer); font-size:10px; text-transform:uppercase; letter-spacing:1px; }}
.top-bar {{ position:fixed; top:0; left:0; right:0; z-index:100; padding:12px 24px; display:flex; justify-content:space-between; align-items:center; background:rgba(13,17,23,0.85); backdrop-filter:blur(8px); border-bottom:1px solid var(--border); }}
.top-bar h1 {{ font-size:16px; font-weight:500; color:var(--text-bright); }}
.hover-linked {{ box-shadow:0 0 10px currentColor; }}
</style>
{DAGRE_CDN}</head><body>
<div class="top-bar"><h1>CEO Cloning — Expansions</h1>{LEGEND_HTML}</div>
<div class="layout" style="padding-top:48px">
    <div class="graph-panel"><svg id="graph"></svg></div>
    <div class="sidebar" id="sidebar">
        <div class="sidebar-inner" id="sidebar-inner"></div>
    </div>
</div>
<script>
{GRAPH_JS}
function showSidebar(cid) {{
    const c = CONCEPTS[cid]; if (!c) return;
    const sb = document.getElementById('sidebar');
    const inner = document.getElementById('sidebar-inner');
    // Build connections
    const inEdges = GRAPH.edges.filter(e => e.to === cid);
    const outEdges = GRAPH.edges.filter(e => e.from === cid);
    let conHtml = '';
    if (inEdges.length) {{
        conHtml += '<strong>From</strong><br>';
        inEdges.forEach(e => {{ const s = CONCEPTS[e.from]; conHtml += (s?s.label:e.from) + ' → this<br>'; }});
    }}
    if (outEdges.length) {{
        conHtml += '<strong>Leads to</strong><br>';
        outEdges.forEach(e => {{ const t = CONCEPTS[e.to]; conHtml += 'this → ' + (t?t.label:e.to) + '<br>'; }});
    }}
    inner.innerHTML = `<button class="sidebar-close" onclick="closeSidebar()">×</button>
        <div class="sidebar-label">${{c.label}}</div>
        <span class="sidebar-layer ${{c.layer}}">${{c.layer}}</span>
        <div class="sidebar-desc">${{c.desc}}</div>
        ${{conHtml ? '<div class="sidebar-connections">' + conHtml + '</div>' : ''}}`;
    sb.classList.add('open');
}}
function closeSidebar() {{ document.getElementById('sidebar').classList.remove('open'); }}
renderGraph('#graph', {{ onClick: showSidebar }});
</script></body></html>"""
(desk / "proto_A_sidebar.html").write_text(a_html)

# ═══════════════════════════════════════════
# PROTOTYPE B: Three columns
# ═══════════════════════════════════════════
print("B: Three columns...")
core_cards = ''.join(f'<div class="card" data-concept="{cid}" onmouseenter="highlightNode(\'{cid}\')" onmouseleave="clearHighlight()"><div class="card-label" style="color:var(--core)">{c["label"]}</div><div class="card-desc">{c["desc"]}</div></div>' for cid, c in CONCEPTS.items() if c["layer"]=="core")
conn_cards = ''.join(f'<div class="card" data-concept="{cid}" onmouseenter="highlightNode(\'{cid}\')" onmouseleave="clearHighlight()"><div class="card-label" style="color:var(--connection)">{c["label"]}</div><div class="card-desc">{c["desc"]}</div></div>' for cid, c in CONCEPTS.items() if c["layer"]=="connection")
front_cards = ''.join(f'<div class="card" data-concept="{cid}" onmouseenter="highlightNode(\'{cid}\')" onmouseleave="clearHighlight()"><div class="card-label" style="color:var(--frontier)">{c["label"]}</div><div class="card-desc">{c["desc"]}</div></div>' for cid, c in CONCEPTS.items() if c["layer"]=="frontier")

b_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Expansions — Three Columns</title>
<style>{SHARED_CSS}
{LEGEND_CSS}
.top-bar {{ position:fixed;top:0;left:0;right:0;z-index:100;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;background:rgba(13,17,23,0.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--border); }}
.top-bar h1 {{ font-size:16px;font-weight:500;color:var(--text-bright); }}
.layout {{ display:flex; height:100vh; padding-top:48px; }}
.col {{ overflow-y:auto; padding:20px; }}
.col-left {{ width:25%; border-right:1px solid var(--border); }}
.col-center {{ flex:1; position:relative; }}
.col-center svg {{ width:100%; height:100%; }}
.col-right {{ width:25%; border-left:1px solid var(--border); }}
.col-header {{ font-size:11px; text-transform:uppercase; letter-spacing:1.5px; color:var(--text-dimmer); margin-bottom:16px; padding-bottom:8px; border-bottom:1px solid var(--border); }}
.card {{ padding:12px 14px; margin-bottom:10px; background:var(--bg-card); border:1px solid var(--border); border-radius:8px; transition:all 0.15s; cursor:default; }}
.card:hover {{ border-color:var(--text-dimmer); }}
.card.highlighted {{ border-color:var(--text-dim); box-shadow:0 0 12px rgba(255,255,255,0.05); }}
.card-label {{ font-size:13px; font-weight:600; margin-bottom:4px; }}
.card-desc {{ font-size:12px; line-height:1.6; color:var(--text-dim); }}
.section-divider {{ font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:var(--text-dimmer); margin:16px 0 8px; }}
</style>
{DAGRE_CDN}</head><body>
<div class="top-bar"><h1>CEO Cloning — Expansions</h1>{LEGEND_HTML}</div>
<div class="layout">
    <div class="col col-left">
        <div class="col-header">Your Ideas</div>
        {core_cards}
    </div>
    <div class="col col-center"><svg id="graph"></svg></div>
    <div class="col col-right">
        <div class="col-header">Connections</div>
        {conn_cards}
        <div class="section-divider" style="margin-top:24px">Frontier</div>
        {front_cards}
    </div>
</div>
<script>
{GRAPH_JS}
function highlightNode(cid) {{
    document.querySelectorAll('.graph-node[data-concept="'+cid+'"]').forEach(n => {{
        n.querySelector('rect,polygon').style.filter = 'drop-shadow(0 0 12px rgba(255,255,255,0.5))';
        n.querySelector('rect,polygon').style.strokeWidth = '3';
    }});
    document.querySelectorAll('.card[data-concept="'+cid+'"]').forEach(c => c.classList.add('highlighted'));
}}
function clearHighlight() {{
    document.querySelectorAll('.graph-node rect, .graph-node polygon').forEach(el => {{
        el.style.filter = ''; el.style.strokeWidth = '';
    }});
    document.querySelectorAll('.card.highlighted').forEach(c => c.classList.remove('highlighted'));
}}
renderGraph('#graph');
</script></body></html>"""
(desk / "proto_B_columns.html").write_text(b_html)

# ═══════════════════════════════════════════
# PROTOTYPE C: Layered reveal
# ═══════════════════════════════════════════
print("C: Layered reveal...")
c_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Expansions — Layered Reveal</title>
<style>{SHARED_CSS}
{LEGEND_CSS}
.top-bar {{ position:fixed;top:0;left:0;right:0;z-index:100;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;background:rgba(13,17,23,0.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--border); }}
.top-bar h1 {{ font-size:16px;font-weight:500;color:var(--text-bright); }}
.controls {{ display:flex; gap:8px; }}
.layer-btn {{ padding:8px 16px; border:1px solid var(--border); border-radius:6px; background:var(--bg); color:var(--text-dim); font-size:13px; cursor:pointer; transition:all 0.2s; }}
.layer-btn.active {{ color:var(--text-bright); border-color:var(--text-dim); }}
.layer-btn.active[data-layer="core"] {{ border-color:var(--core); color:var(--core); }}
.layer-btn.active[data-layer="connection"] {{ border-color:var(--connection); color:var(--connection); }}
.layer-btn.active[data-layer="frontier"] {{ border-color:var(--frontier); color:var(--frontier); }}
.graph-wrap {{ width:100vw; height:calc(100vh - 48px); margin-top:48px; }}
.graph-wrap svg {{ width:100%; height:100%; }}
.tooltip {{ position:fixed; z-index:200; background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:12px 16px; max-width:300px; pointer-events:none; opacity:0; transition:opacity 0.15s; box-shadow:0 4px 16px rgba(0,0,0,0.4); }}
.tooltip.visible {{ opacity:1; }}
.tooltip-label {{ font-size:14px; font-weight:600; color:var(--text-bright); margin-bottom:4px; }}
.tooltip-desc {{ font-size:12px; line-height:1.6; color:var(--text-dim); }}
</style>
{DAGRE_CDN}</head><body>
<div class="top-bar">
    <h1>CEO Cloning — Expansions</h1>
    <div class="controls">
        <button class="layer-btn active" data-layer="core" onclick="toggleLayer('core',this)">Your Ideas</button>
        <button class="layer-btn" data-layer="connection" onclick="toggleLayer('connection',this)">+ Connections</button>
        <button class="layer-btn" data-layer="frontier" onclick="toggleLayer('frontier',this)">+ Frontier</button>
    </div>
    {LEGEND_HTML}
</div>
<div class="graph-wrap"><svg id="graph"></svg></div>
<div class="tooltip" id="tooltip"><div class="tooltip-label" id="tt-label"></div><div class="tooltip-desc" id="tt-desc"></div></div>
<script>
{GRAPH_JS}
let activeLayers = ['core'];
function toggleLayer(layer, btn) {{
    if (layer === 'core') return; // always on
    btn.classList.toggle('active');
    if (btn.classList.contains('active')) {{
        // Add this layer and all layers before it
        if (layer === 'connection') activeLayers = ['core','connection'];
        if (layer === 'frontier') {{
            activeLayers = ['core','connection','frontier'];
            document.querySelector('[data-layer="connection"]').classList.add('active');
        }}
    }} else {{
        // Remove this and layers after
        if (layer === 'connection') {{
            activeLayers = ['core'];
            document.querySelector('[data-layer="frontier"]').classList.remove('active');
        }}
        if (layer === 'frontier') activeLayers = activeLayers.filter(l => l !== 'frontier');
    }}
    renderGraph('#graph', {{
        layers: activeLayers,
        onHover: showTooltip,
        onLeave: hideTooltip
    }});
}}
const tt = document.getElementById('tooltip');
function showTooltip(cid) {{
    const c = CONCEPTS[cid]; if (!c) return;
    document.getElementById('tt-label').textContent = c.label;
    document.getElementById('tt-desc').textContent = c.desc;
    tt.classList.add('visible');
}}
function hideTooltip() {{ tt.classList.remove('visible'); }}
document.addEventListener('mousemove', e => {{
    tt.style.left = (e.clientX + 16) + 'px';
    tt.style.top = (e.clientY + 16) + 'px';
}});
renderGraph('#graph', {{ layers: activeLayers, onHover: showTooltip, onLeave: hideTooltip }});
</script></body></html>"""
(desk / "proto_C_layers.html").write_text(c_html)

# ═══════════════════════════════════════════
# PROTOTYPE D: Annotated graph (everything in tooltips)
# ═══════════════════════════════════════════
print("D: Annotated graph...")
d_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Expansions — Annotated Graph</title>
<style>{SHARED_CSS}
{LEGEND_CSS}
.top-bar {{ position:fixed;top:0;left:0;right:0;z-index:100;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;background:rgba(13,17,23,0.9);backdrop-filter:blur(8px);border-bottom:1px solid var(--border); }}
.top-bar h1 {{ font-size:16px;font-weight:500;color:var(--text-bright); }}
.graph-wrap {{ width:100vw; height:calc(100vh - 48px); margin-top:48px; }}
.graph-wrap svg {{ width:100%; height:100%; }}
.popover {{ position:fixed; z-index:200; background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:16px 20px; max-width:340px; box-shadow:0 8px 32px rgba(0,0,0,0.5); opacity:0; transform:translateY(8px); transition:all 0.2s; pointer-events:none; }}
.popover.visible {{ opacity:1; transform:translateY(0); pointer-events:auto; }}
.popover-label {{ font-size:15px; font-weight:600; color:var(--text-bright); margin-bottom:4px; }}
.popover-layer {{ font-size:11px; text-transform:uppercase; letter-spacing:1px; padding:2px 8px; border-radius:4px; display:inline-block; margin-bottom:10px; }}
.popover-layer.core {{ background:rgba(86,180,233,0.15); color:var(--core); }}
.popover-layer.connection {{ background:rgba(230,159,0,0.15); color:var(--connection); }}
.popover-layer.frontier {{ background:rgba(204,121,167,0.15); color:var(--frontier); }}
.popover-desc {{ font-size:13px; line-height:1.7; color:var(--text); }}
.popover-edges {{ margin-top:10px; font-size:11px; color:var(--text-dim); line-height:1.6; }}
</style>
{DAGRE_CDN}</head><body>
<div class="top-bar"><h1>CEO Cloning — Expansions</h1>{LEGEND_HTML}</div>
<div class="graph-wrap"><svg id="graph"></svg></div>
<div class="popover" id="popover">
    <div class="popover-label" id="pop-label"></div>
    <span class="popover-layer" id="pop-layer"></span>
    <div class="popover-desc" id="pop-desc"></div>
    <div class="popover-edges" id="pop-edges"></div>
</div>
<script>
{GRAPH_JS}
const pop = document.getElementById('popover');
let popVisible = false, popCid = null;

function showPop(cid) {{
    const c = CONCEPTS[cid]; if (!c) return;
    popCid = cid;
    document.getElementById('pop-label').textContent = c.label;
    const layerEl = document.getElementById('pop-layer');
    layerEl.textContent = c.layer;
    layerEl.className = 'popover-layer ' + c.layer;
    document.getElementById('pop-desc').textContent = c.desc;
    // Edges
    const inE = GRAPH.edges.filter(e => e.to === cid);
    const outE = GRAPH.edges.filter(e => e.from === cid);
    let eHtml = '';
    inE.forEach(e => {{ const s = CONCEPTS[e.from]; eHtml += '← ' + (s?s.label:e.from) + '<br>'; }});
    outE.forEach(e => {{ const t = CONCEPTS[e.to]; eHtml += '→ ' + (t?t.label:e.to) + '<br>'; }});
    document.getElementById('pop-edges').innerHTML = eHtml;
    pop.classList.add('visible');
    popVisible = true;
}}

renderGraph('#graph', {{
    onClick: (cid) => {{
        if (popVisible && popCid === cid) {{
            pop.classList.remove('visible');
            popVisible = false;
        }} else {{
            showPop(cid);
        }}
    }}
}});

document.addEventListener('mousemove', e => {{
    if (popVisible) {{
        const x = Math.min(e.clientX + 16, window.innerWidth - 360);
        const y = Math.min(e.clientY + 16, window.innerHeight - 200);
        pop.style.left = x + 'px';
        pop.style.top = y + 'px';
    }}
}});
document.addEventListener('click', e => {{
    if (popVisible && !e.target.closest('.graph-node') && !e.target.closest('.popover')) {{
        pop.classList.remove('visible');
        popVisible = false;
    }}
}});
</script></body></html>"""
(desk / "proto_D_annotated.html").write_text(d_html)

print("Done! Four prototypes on Desktop:")
print("  proto_A_sidebar.html  — Graph + click sidebar")
print("  proto_B_columns.html  — Three columns")
print("  proto_C_layers.html   — Layered reveal (toggle)")
print("  proto_D_annotated.html — Annotated graph (click popovers)")
