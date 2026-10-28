"""
Multimodal Construction Intelligence Platform
Skill Wallet Capstone Project 1

Streamlit app: PPE audit, crack diagnostics, planning & valuation, weather-aware site assistant.
Runs with a Google Gemini API key, or in demonstration mode without one.
No database required -> 1-click deploy on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime
from typing import Any

import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai

    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    GENAI_AVAILABLE = False

MODEL_NAME = "gemini-1.5-flash"

# --------------------------------------------------------------------------------------
# Page config & styling
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
.stApp { background: linear-gradient(180deg, #0f1419 0%, #151b22 100%); color: #e8e6e1; }
section[data-testid="stSidebar"] { background: #0b0f13; border-right: 1px solid #2a323c; }
.hero { padding: 1.4rem 1.6rem; border-radius: 14px; border: 1px solid #2a323c;
  background: linear-gradient(120deg, rgba(255,176,32,.14), rgba(255,176,32,0) 60%), #121820; margin-bottom: 1rem; }
.hero h1 { margin: 0; font-size: 2rem; color: #ffb020; }
.hero p { margin: .3rem 0 0; color: #9aa4af; }
.metric-card { background: #121820; border: 1px solid #2a323c; border-radius: 12px; padding: 1rem 1.2rem; height: 100%; }
.metric-card .label { color: #9aa4af; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.metric-card .value { font-family: 'Space Grotesk'; font-size: 1.7rem; font-weight: 700; color: #f5f3ee; }
.metric-card .sub { color: #9aa4af; font-size: .8rem; }
.pill { display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .75rem; font-weight: 600; margin: 0 .25rem .25rem 0; }
.pill-ok { background: rgba(46,204,113,.15); color: #2ecc71; border: 1px solid rgba(46,204,113,.4); }
.pill-low { background: rgba(52,152,219,.15); color: #5dade2; border: 1px solid rgba(52,152,219,.4); }
.pill-med { background: rgba(255,176,32,.15); color: #ffb020; border: 1px solid rgba(255,176,32,.4); }
.pill-high { background: rgba(230,126,34,.18); color: #f39c12; border: 1px solid rgba(230,126,34,.5); }
.pill-crit { background: rgba(231,76,60,.18); color: #ff6b5b; border: 1px solid rgba(231,76,60,.5); }
.hazard { background: #121820; border-left: 4px solid #ffb020; border-radius: 8px; padding: .8rem 1rem; margin-bottom: .6rem; }
.alert { border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .6rem; border: 1px solid; }
.alert-crit { background: rgba(231,76,60,.1); border-color: rgba(231,76,60,.5); }
.alert-warn { background: rgba(255,176,32,.08); border-color: rgba(255,176,32,.45); }
.alert-ok { background: rgba(46,204,113,.08); border-color: rgba(46,204,113,.4); }
.demo-banner { background: rgba(255,176,32,.1); border: 1px dashed #ffb020; color: #ffb020;
  padding: .5rem .9rem; border-radius: 8px; font-size: .85rem; margin-bottom: .8rem; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Grotesk'; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #ffb020 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SEVERITY_PILL = {
    "low": "pill-low",
    "medium": "pill-med",
    "high": "pill-high",
    "critical": "pill-crit",
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{text}</span>'


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
    )


def format_inr(amount: float) -> str:
    """Format rupees as Lakhs / Crores."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    if amount >= 1e5:
        return f"₹{amount / 1e5:,.2f} L"
    return f"₹{amount:,.0f}"


def extract_json(text: str) -> dict[str, Any]:
    """Robustly pull a JSON object out of a model response."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    return json.loads(text)


def get_model(api_key: str | None, json_mode: bool = False):
    if not (api_key and GENAI_AVAILABLE):
        return None
    genai.configure(api_key=api_key)
    cfg = {"temperature": 0.2}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    return genai.GenerativeModel(MODEL_NAME, generation_config=cfg)


def run_vision(api_key: str | None, prompt: str, image: Image.Image, demo: dict) -> tuple[dict, bool]:
    """Returns (result, is_demo)."""
    model = get_model(api_key, json_mode=True)
    if model is None:
        return demo, True
    try:
        resp = model.generate_content([prompt, image])
        return extract_json(resp.text), False
    except json.JSONDecodeError:
        st.error("The AI returned an unreadable response. Showing demonstration output instead.")
    except Exception as exc:  # network, auth, quota, safety block
        st.error(f"Gemini request failed: {exc}. Showing demonstration output instead.")
    return demo, True


def download_json(data: dict, name: str) -> None:
    st.download_button(
        "⬇️ Download JSON report",
        data=json.dumps(data, indent=2),
        file_name=f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------------------------------------------
# Prompts & demo data
# --------------------------------------------------------------------------------------
PPE_PROMPT = """You are a certified construction safety auditor (OSHA 29 CFR 1926 and Indian codes
including IS 456 practices and IS 3764 safety). Analyse this construction site photo.
Assess PPE compliance for all visible workers: helmets, high-visibility vests, steel-toe boots,
safety harnesses (where working at height). Identify every site hazard.
Return ONLY JSON with this exact schema:
{
 "safety_score": int 0-100,
 "overall_status": "Compliant" | "Partially Compliant" | "Non-Compliant",
 "workers_detected": int,
 "ppe_detection": {
   "helmets": {"status": "Detected"|"Missing"|"Partial"|"Not Visible", "notes": str},
   "high_vis_vests": {...same},
   "steel_toe_boots": {...same},
   "safety_harnesses": {...same}
 },
 "hazards": [{"hazard": str, "severity": "Low"|"Medium"|"High"|"Critical",
              "standard_reference": str, "corrective_remedy": str}],
 "summary": str
}"""

PPE_DEMO = {
    "safety_score": 62,
    "overall_status": "Partially Compliant",
    "workers_detected": 5,
    "ppe_detection": {
        "helmets": {"status": "Partial", "notes": "4 of 5 workers wearing hard hats."},
        "high_vis_vests": {"status": "Detected", "notes": "All visible workers wear Class 2 vests."},
        "steel_toe_boots": {"status": "Not Visible", "notes": "Footwear obscured by debris."},
        "safety_harnesses": {"status": "Missing", "notes": "Worker on scaffold level 2 lacks fall arrest."},
    },
    "hazards": [
        {
            "hazard": "Work at height without fall protection",
            "severity": "Critical",
            "standard_reference": "OSHA 1926.501(b)",
            "corrective_remedy": "Stop work; provide full-body harness anchored to rated lifeline.",
        },
        {
            "hazard": "Unbarricaded rebar starter bars",
            "severity": "High",
            "standard_reference": "OSHA 1926.701(b)",
            "corrective_remedy": "Fit mushroom caps on all exposed rebar ends.",
        },
        {
            "hazard": "Loose material stacking near edge",
            "severity": "Medium",
            "standard_reference": "OSHA 1926.250",
            "corrective_remedy": "Restack ≥1.8 m from slab edge; install toe boards.",
        },
    ],
    "summary": "Demonstration result: fall protection gap is the dominant risk.",
}

CRACK_PROMPT = """You are a senior structural engineer. Analyse this concrete / masonry surface.
Classify the defect as one of: Hairline, Settlement, Structural Shear, Thermal Expansion, Spalling
(or "No Defect"). Estimate crack width in mm. Reference IS 456:2000 crack-width limits (0.3 mm
moderate exposure, 0.2 mm severe, 0.1 mm aggressive).
Return ONLY JSON:
{
 "defect_type": str,
 "confidence": int 0-100,
 "estimated_crack_width_mm": number,
 "crack_pattern": str,
 "root_causes": [str],
 "recommended_repairs": [{"method": str, "description": str}],
 "urgency": "Monitor" | "Low" | "Medium" | "High" | "Immediate",
 "structural_risk": str,
 "is456_compliance": str
}"""

CRACK_DEMO = {
    "defect_type": "Settlement",
    "confidence": 78,
    "estimated_crack_width_mm": 1.8,
    "crack_pattern": "Diagonal stepped crack from window corner toward base.",
    "root_causes": [
        "Differential foundation settlement",
        "Inadequate soil compaction beneath strip footing",
        "Seasonal moisture variation in expansive clay",
    ],
    "recommended_repairs": [
        {"method": "Epoxy injection", "description": "Low-viscosity epoxy at 0.5 MPa through surface ports."},
        {"method": "Structural re-grouting", "description": "Pressure-grout voids beneath footing; consider micro-piles."},
        {"method": "Crack monitoring", "description": "Install tell-tale gauges; log weekly for 3 months."},
    ],
    "urgency": "High",
    "structural_risk": "Exceeds IS 456 0.3 mm limit; progressive settlement possible.",
    "is456_compliance": "Non-compliant (width > 0.3 mm)",
}

ASSISTANT_SYSTEM = """You are SiteMind, an expert construction engineering assistant for Indian and
international sites. You know IS 456, IS 800, IS 1893, NBC 2016, OSHA 1926, concrete technology,
project planning, quantity estimation, and site safety. Be concise, practical, and cite codes where
relevant. Current site weather condition: {weather}."""


def _secret_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏗️ Control Center")
    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        value=_secret_key(),
        help="Get one at aistudio.google.com. Leave blank for demonstration mode.",
    ).strip() or None

    if api_key and GENAI_AVAILABLE:
        st.success(f"Live AI mode · {MODEL_NAME}")
    elif api_key and not GENAI_AVAILABLE:
        st.error("google-generativeai not installed. Running in demo mode.")
        api_key = None
    else:
        st.info("Demonstration mode — sample outputs shown.")

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown(
        """
```
Streamlit UI
  ├─ Vision (Gemini multimodal)
  │   ├─ PPE Audit  → JSON
  │   └─ Crack Dx   → JSON
  ├─ Planning engine (local)
  │   └─ Pollinations render
  └─ Assistant (Gemini chat)
      └─ Weather rules engine
Session state · No DB
```
"""
    )
    st.markdown("### Modules")
    st.markdown(
        "1. 🦺 Site Safety & PPE Audit\n2. 🧱 Crack Diagnostics\n"
        "3. 📐 Planning & Valuation\n4. 🌦️ Site Assistant"
    )
    st.caption("Skill Wallet Capstone Project 1")

DEMO = api_key is None

st.markdown(
    '<div class="hero"><h1>Multimodal Construction Intelligence Platform</h1>'
    "<p>AI-powered safety audits, structural diagnostics, project valuation and site advisory.</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🦺 Safety & PPE Audit", "🧱 Crack Diagnostics", "📐 Planning & Valuation", "🌦️ Site Assistant"]
)

# --------------------------------------------------------------------------------------
# Tab 1 — PPE audit
# --------------------------------------------------------------------------------------
with tab1:
    st.subheader("Site Safety & PPE Compliance Audit")
    st.caption("OSHA 1926 · IS 456 / IS 3764 site practice")
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="ppe")
    if up:
        try:
            img = Image.open(up).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read image: {exc}")
            img = None
        if img:
            c1, c2 = st.columns([1, 1.4])
            c1.image(img, use_container_width=True)
            with c2:
                if st.button("Run PPE audit", type="primary", use_container_width=True):
                    with st.spinner("Analysing site photo…"):
                        st.session_state.ppe_result, st.session_state.ppe_demo = run_vision(
                            api_key, PPE_PROMPT, img, PPE_DEMO
                        )

            res = st.session_state.get("ppe_result")
            if res:
                if st.session_state.get("ppe_demo"):
                    st.markdown('<div class="demo-banner">Demonstration output — add an API key for live analysis.</div>', unsafe_allow_html=True)
                score = int(res.get("safety_score", 0))
                status = res.get("overall_status", "Unknown")
                hazards = res.get("hazards", [])
                crit = sum(1 for h in hazards if str(h.get("severity", "")).lower() == "critical")
                m = st.columns(4)
                m[0].markdown(metric_card("Safety score", f"{score}/100"), unsafe_allow_html=True)
                m[1].markdown(metric_card("Status", status), unsafe_allow_html=True)
                m[2].markdown(metric_card("Workers", str(res.get("workers_detected", "—"))), unsafe_allow_html=True)
                m[3].markdown(metric_card("Hazards", str(len(hazards)), f"{crit} critical"), unsafe_allow_html=True)
                st.progress(min(max(score, 0), 100) / 100)

                st.markdown("#### PPE checklist")
                ppe = res.get("ppe_detection", {})
                cols = st.columns(4)
                for col, (k, v) in zip(cols, ppe.items()):
                    v = v if isinstance(v, dict) else {"status": str(v), "notes": ""}
                    s = v.get("status", "Unknown")
                    cls = {"Detected": "pill-ok", "Partial": "pill-med", "Missing": "pill-crit"}.get(s, "pill-low")
                    col.markdown(
                        f'<div class="metric-card"><div class="label">{k.replace("_", " ")}</div>'
                        f'{pill(s, cls)}<div class="sub">{v.get("notes", "")}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Identified hazards")
                for h in hazards:
                    sev = str(h.get("severity", "Low"))
                    st.markdown(
                        f'<div class="hazard"><b>{h.get("hazard", "")}</b> '
                        f'{pill(sev, SEVERITY_PILL.get(sev.lower(), "pill-low"))}'
                        f'<div class="sub" style="color:#9aa4af">{h.get("standard_reference", "")}</div>'
                        f'<div>🔧 {h.get("corrective_remedy", "")}</div></div>',
                        unsafe_allow_html=True,
                    )
                if res.get("summary"):
                    st.info(res["summary"])
                download_json(res, "ppe_audit")
    else:
        st.session_state.pop("ppe_result", None)

# --------------------------------------------------------------------------------------
# Tab 2 — Crack diagnostics
# --------------------------------------------------------------------------------------
with tab2:
    st.subheader("Crack Diagnostics & Structural Integrity")
    st.caption("IS 456:2000 crack-width limits · repair method selection")
    up2 = st.file_uploader("Upload a concrete / wall surface photo", type=["jpg", "jpeg", "png"], key="crack")
    if up2:
        try:
            img2 = Image.open(up2).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read image: {exc}")
            img2 = None
        if img2:
            c1, c2 = st.columns([1, 1.4])
            c1.image(img2, use_container_width=True)
            with c2:
                if st.button("Diagnose surface", type="primary", use_container_width=True):
                    with st.spinner("Inspecting defect…"):
                        st.session_state.crack_result, st.session_state.crack_demo = run_vision(
                            api_key, CRACK_PROMPT, img2, CRACK_DEMO
                        )
            r = st.session_state.get("crack_result")
            if r:
                if st.session_state.get("crack_demo"):
                    st.markdown('<div class="demo-banner">Demonstration output — add an API key for live analysis.</div>', unsafe_allow_html=True)
                urg = str(r.get("urgency", "Monitor"))
                ucls = {"Immediate": "pill-crit", "High": "pill-high", "Medium": "pill-med"}.get(urg, "pill-low")
                m = st.columns(4)
                m[0].markdown(metric_card("Defect type", r.get("defect_type", "—")), unsafe_allow_html=True)
                m[1].markdown(metric_card("Est. width", f'{r.get("estimated_crack_width_mm", 0)} mm', "IS 456 limit 0.3 mm"), unsafe_allow_html=True)
                m[2].markdown(metric_card("Confidence", f'{r.get("confidence", 0)}%'), unsafe_allow_html=True)
                m[3].markdown(f'<div class="metric-card"><div class="label">Urgency</div>{pill(urg, ucls)}</div>', unsafe_allow_html=True)

                a, b = st.columns(2)
                with a:
                    st.markdown("#### Root causes")
                    for c in r.get("root_causes", []):
                        st.markdown(f"- {c}")
                    st.markdown("#### Structural risk")
                    st.write(r.get("structural_risk", ""))
                    st.caption(f'IS 456: {r.get("is456_compliance", "")}')
                with b:
                    st.markdown("#### Recommended repairs")
                    for rep in r.get("recommended_repairs", []):
                        if isinstance(rep, dict):
                            st.markdown(f'**{rep.get("method", "")}** — {rep.get("description", "")}')
                        else:
                            st.markdown(f"- {rep}")
                download_json(r, "crack_diagnostics")
    else:
        st.session_state.pop("crack_result", None)

# --------------------------------------------------------------------------------------
# Tab 3 — Planning & valuation
# --------------------------------------------------------------------------------------
STRUCTURES = {
    # name: (construction cost ₹/sqft base, steel kg/sqft, cement bags/sqft, market multiplier)
    "Residential Villa": (2200, 4.0, 0.40, 1.45),
    "Apartment Tower (G+10)": (2600, 4.5, 0.45, 1.60),
    "Commercial Office": (3000, 5.0, 0.50, 1.75),
    "Warehouse / Industrial": (1600, 3.5, 0.30, 1.25),
    "Retail Mall": (3400, 5.2, 0.52, 1.85),
}
TIERS = {
    "Tier 1 (Mumbai, Delhi, Bengaluru…)": (1.35, 1.9),
    "Tier 2 (Pune, Jaipur, Lucknow…)": (1.10, 1.3),
    "Tier 3 (Smaller cities)": (0.95, 1.0),
}
MILESTONES = [(10, "Excavation & foundation"), (35, "Superstructure frame"), (55, "Masonry & roofing"),
              (75, "MEP services"), (90, "Finishing"), (100, "Handover")]


def compute_valuation(structure: str, sqft: float, tier: str, pct: float) -> dict:
    cost, steel, cement, mult = STRUCTURES[structure]
    cost_f, land_f = TIERS[tier]
    budget = sqft * cost * cost_f
    capex = budget * pct / 100
    # Market value: land+premium component scaled by tier, realised progressively with completion
    completed_value = budget * mult * land_f
    valuation = capex * 1.05 + (completed_value - budget) * (pct / 100) ** 1.5
    return {
        "budget": budget,
        "capex": capex,
        "remaining": budget - capex,
        "valuation": valuation,
        "completed_value": completed_value,
        "steel_t": sqft * steel / 1000,
        "cement_bags": sqft * cement,
        "steel_used_t": sqft * steel / 1000 * pct / 100,
        "cement_used": sqft * cement * pct / 100,
    }


with tab3:
    st.subheader("Planning & Real Estate Valuation")
    i1, i2 = st.columns(2)
    with i1:
        structure = st.selectbox("Structure type", list(STRUCTURES))
        sqft = st.number_input("Built-up area (sqft)", 200, 2_000_000, 2500, step=100)
    with i2:
        tier = st.selectbox("Location tier", list(TIERS))
        pct = st.slider("Completion %", 0, 100, 40)

    v = compute_valuation(structure, float(sqft), tier, float(pct))
    m = st.columns(4)
    m[0].markdown(metric_card("Total budget", format_inr(v["budget"])), unsafe_allow_html=True)
    m[1].markdown(metric_card("CapEx incurred", format_inr(v["capex"]), f'{format_inr(v["remaining"])} remaining'), unsafe_allow_html=True)
    m[2].markdown(metric_card("Current valuation", format_inr(v["valuation"]), f'At completion: {format_inr(v["completed_value"])}'), unsafe_allow_html=True)
    m[3].markdown(metric_card("Materials", f'{v["steel_t"]:,.1f} t steel', f'{v["cement_bags"]:,.0f} cement bags'), unsafe_allow_html=True)

    st.markdown("#### Milestone progress")
    st.progress(pct / 100, text=f"{pct}% complete")
    ms_cols = st.columns(len(MILESTONES))
    for col, (th, name) in zip(ms_cols, MILESTONES):
        col.markdown(pill(f"{'✓' if pct >= th else '○'} {name}", "pill-ok" if pct >= th else "pill-low"), unsafe_allow_html=True)

    st.markdown("#### Material consumption to date")
    mc = st.columns(2)
    mc[0].metric("Steel consumed", f'{v["steel_used_t"]:,.1f} t', f'of {v["steel_t"]:,.1f} t')
    mc[1].metric("Cement consumed", f'{v["cement_used"]:,.0f} bags', f'of {v["cement_bags"]:,.0f} bags')

    st.markdown("#### Conceptual 3D architectural rendering")
    style = st.selectbox("Architectural style", ["Modern minimalist", "Contemporary Indian", "Glass façade high-rise", "Brutalist concrete", "Sustainable green"])
    render_prompt = (
        f"photorealistic 3D architectural render of a {style.lower()} {structure.lower()}, "
        f"{int(sqft)} sqft, located in an Indian {tier.split(' (')[0]} city, golden hour, "
        "landscaped surroundings, high detail, architectural visualization"
    )
    st.code(render_prompt, language="text")
    if st.button("Generate rendering", use_container_width=True):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(render_prompt)}?width=1024&height=576&nologo=true"
        try:
            st.image(url, caption="Pollinations AI conceptual render", use_container_width=True)
        except Exception as exc:
            st.error(f"Rendering service unavailable: {exc}")

    download_json({"inputs": {"structure": structure, "sqft": sqft, "tier": tier, "completion_pct": pct},
                   "results": {k: round(x, 2) for k, x in v.items()}}, "valuation")

# --------------------------------------------------------------------------------------
# Tab 4 — Weather advisory & assistant
# --------------------------------------------------------------------------------------
WEATHER_ALERTS = {
    "High Wind": [
        ("crit", "Suspend tower crane operations", "Stop lifting when sustained wind > 38 km/h (IS 13367 / manufacturer limit)."),
        ("crit", "Secure work at height", "Halt scaffold & roof work; tie down sheeting and loose materials."),
        ("warn", "Formwork inspection", "Check props and bracing on vertical formwork before re-entry."),
    ],
    "Heavy Rain": [
        ("crit", "Electrical hazard mitigation", "Isolate temporary DBs, use ELCB/RCCB, keep cables off wet ground."),
        ("crit", "Excavation stability", "Inspect trench shoring; dewater pits; no entry if walls are saturated."),
        ("warn", "Concreting", "Postpone open pours; cover fresh concrete with polythene."),
    ],
    "Extreme Heat": [
        ("crit", "Heat-stress protocol", "Shift heavy work to 6–11 am; mandatory shade & water breaks every hour."),
        ("warn", "Concrete curing protocol", "Keep placing temp < 35 °C (IS 7861 Pt 1); use chilled water, cure ≥ 10 days with wet hessian."),
        ("warn", "Equipment overheating", "Monitor diesel generators and compressors."),
    ],
    "Normal": [
        ("ok", "All operations permitted", "Standard toolbox talk and PPE checks apply."),
        ("ok", "Curing", "Moist cure ≥ 7 days for OPC concrete (IS 456 cl. 13.5)."),
    ],
}

DEMO_REPLIES = [
    "For M25 concrete in a slab, a typical nominal mix is 1:1:2 by volume, but IS 456 recommends design mix (IS 10262). Maintain w/c ≤ 0.50 for moderate exposure and cure for at least 7 days.",
    "Minimum clear cover per IS 456 Table 16: slabs 20 mm, beams 25 mm, columns 40 mm, footings 50 mm (moderate exposure).",
    "In demonstration mode I give sample guidance. Add a Gemini API key in the sidebar for full conversational answers.",
]

with tab4:
    st.subheader("Daily Site Assistant & Weather-Aware Advisory")
    weather = st.radio("Current site weather", list(WEATHER_ALERTS), horizontal=True, index=3)
    for lvl, title, body in WEATHER_ALERTS[weather]:
        icon = {"crit": "⛔", "warn": "⚠️", "ok": "✅"}[lvl]
        st.markdown(f'<div class="alert alert-{lvl}"><b>{icon} {title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.markdown("#### SiteMind assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if st.button("Clear conversation"):
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar="👷" if msg["role"] == "user" else "🏗️"):
            st.markdown(msg["content"])

    if q := st.chat_input("Ask about concrete, codes, scheduling, safety…"):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user", avatar="👷"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🏗️"):
            reply = None
            if not DEMO:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(MODEL_NAME, system_instruction=ASSISTANT_SYSTEM.format(weather=weather))
                    history = [
                        {"role": "user" if m_["role"] == "user" else "model", "parts": [m_["content"]]}
                        for m_ in st.session_state.chat[:-1]
                    ]
                    chat = model.start_chat(history=history)
                    stream = chat.send_message(q, stream=True)
                    reply = st.write_stream(chunk.text for chunk in stream if getattr(chunk, "text", None))
                except Exception as exc:
                    st.error(f"Assistant error: {exc}")
            if not reply:
                reply = DEMO_REPLIES[len(st.session_state.chat) // 2 % len(DEMO_REPLIES)]
                st.markdown(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

st.markdown("---")
st.caption("AI outputs are advisory and must be verified by a qualified engineer or safety officer.")
