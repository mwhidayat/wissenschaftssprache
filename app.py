import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AkademiKorpus – Wissenschaftssprache Deutsch",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --primary:   #1a3a5c;
    --accent:    #c8541e;
    --surface:   #f4f1eb;
    --border:    #d4c9b8;
    --text:      #1e1e1e;
    --muted:     #6b6357;
    --tag-bg:    #e8e3da;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--primary) !important;
    border-right: 3px solid #0f2540;
}
section[data-testid="stSidebar"] * {
    color: #e8ddd0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    color: #b8cde0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px;
    font-weight: 600;
    width: 100%;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'IBM Plex Serif', serif !important;
}

/* Main area */
.main > div {
    padding-top: 1.5rem;
}

/* Header */
.corpus-header {
    border-bottom: 3px solid var(--primary);
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
}
.corpus-header h1 {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--primary);
    margin: 0 0 0.2rem 0;
}
.corpus-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1.25rem;
    flex: 1;
    min-width: 120px;
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: var(--primary);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* Result card */
.result-card {
    background: #ffffff;
    border: 1.5px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 6px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
}
.result-card .meta {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.5rem;
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
}
.result-card .meta .badge {
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-size: 0.68rem;
    color: var(--primary);
    font-weight: 600;
    letter-spacing: 0.04em;
}
.result-card .meta .score-badge {
    background: var(--primary);
    color: white;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-size: 0.68rem;
    font-weight: 600;
}
.result-card h4 {
    font-family: 'IBM Plex Serif', serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--primary);
    margin: 0 0 0.5rem 0;
}
.result-card .body-text {
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--text);
    margin-bottom: 0.6rem;
}
.result-card .ann-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin-top: 0.5rem;
}
.result-card .ann-block .ann-label {
    display: inline-block;
    background: var(--primary);
    color: #fff;
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    margin-right: 0.4rem;
    margin-bottom: 0.2rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.result-card .ann-span {
    font-size: 0.82rem;
    color: var(--muted);
    font-style: italic;
    display: block;
    margin-bottom: 0.35rem;
    padding-left: 0.25rem;
    border-left: 2px solid var(--border);
}

/* Annotation filter tags */
.label-pill {
    display: inline-block;
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.75rem;
    margin: 0.15rem;
    cursor: default;
    color: var(--primary);
    font-weight: 500;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 2px solid var(--border);
    margin: 1.5rem 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--muted);
    font-family: 'IBM Plex Serif', serif;
    font-style: italic;
    font-size: 1rem;
}

/* Tab style override */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Highlight query match */
mark {
    background: #fde68a;
    border-radius: 2px;
    padding: 0.05rem 0.1rem;
}

.stExpander {
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_JSON_PATH = Path("de_akademie.json")

LABEL_COLORS = {
    "definition": "#1a3a5c",
    "research_question": "#7c3aed",
    "problem_statement": "#b91c1c",
    "argument": "#1e6b3c",
    "citation": "#92400e",
    "paraphrase": "#c87000",
    "contrast": "#0e6b75",
    "comparison": "#0e6b75",
    "data_description": "#1e5a8c",
    "method_description": "#2d5a1e",
    "result_statement": "#6b21a8",
    "passive_style": "#64748b",
    "limitation": "#dc2626",
    "method": "#2d5a1e",
}

LABEL_DESCRIPTIONS = {
    "definition": "Begriffsdefinition / Erläuterung",
    "research_question": "Forschungsfrage / Fragestellung",
    "problem_statement": "Problemstellung / Problematik",
    "argument": "Argument / Begründung",
    "citation": "Zitat / Quellenangabe",
    "paraphrase": "Paraphrase (Konjunktiv I)",
    "contrast": "Kontrast / Gegensatz",
    "comparison": "Vergleich / Ähnlichkeit",
    "data_description": "Datenbeschreibung / Statistik",
    "method_description": "Methodenbeschreibung",
    "result_statement": "Ergebnisaussage",
    "passive_style": "Passivkonstruktion / Nominalstil",
    "limitation": "Einschränkung",
    "method": "Methode",
}

WEEK_TOPICS = {
    1: "Alltägliche Wissenschaftssprache",
    2: "Begriffserläuterung und Definition",
    3: "Frage, Problem und Verwandtes",
    4: "Thematisierung, Kommentierung und Gliederung",
    5: "Beziehungen und Verweise im Text",
    6: "Argumentieren und Argumentation",
    9: "Gegenüberstellung und Vergleich",
    10: "Quantitäten und Grafikbeschreibung",
    11: "Zitat, Wiedergabe und Paraphrase",
    12: "Sachlicher Stil, Aktiv und Passiv",
}

DIFFICULTY_COLOR = {"leicht": "#1e6b3c", "mittel": "#c87000", "schwer": "#b91c1c"}

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_corpus(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def corpus_to_dataframe(json_path: str) -> pd.DataFrame:
    data = load_corpus(json_path)
    rows = []
    for doc in data.get("documents", []):
        base = {
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "domain": doc.get("domain", ""),
            "topic": doc.get("topic", ""),
            "course_week": doc.get("course_week", 0),
            "difficulty": doc.get("difficulty", ""),
            "text_type": doc.get("text_type", ""),
            "course_target": doc.get("course_target", ""),
        }
        for section_obj in doc.get("sections", []):
            section_name = section_obj.get("section", "")
            for para_idx, para in enumerate(section_obj.get("paragraphs", []), 1):
                annotations = para.get("annotations", []) or []
                labels = [a.get("label", "") for a in annotations if isinstance(a, dict)]
                spans = [a.get("span", "") for a in annotations if isinstance(a, dict)]
                rows.append(
                    {
                        **base,
                        "section": section_name,
                        "paragraph_index": para_idx,
                        "text": para.get("text", ""),
                        "annotated_text": para.get("annotated_text", ""),
                        "annotations": annotations,
                        "annotation_labels": labels,
                        "annotation_spans": spans,
                        "annotation_label_text": " | ".join(sorted(set(labels))),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["search_blob"] = (
            df[["doc_id", "title", "domain", "topic", "section", "text", "annotated_text", "annotation_label_text"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
    else:
        df["search_blob"] = []
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────
def tokenize_query(query: str) -> list:
    return [t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if t]


def score_row(row: pd.Series, terms: list) -> int:
    blob = row.get("search_blob", "")
    return sum(blob.count(t) for t in terms) if isinstance(blob, str) else 0


def highlight_text(text: str, terms: list) -> str:
    if not terms:
        return text
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)


def label_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#444")
    return f'<span style="display:inline-block;background:{color};color:#fff;font-size:0.65rem;font-family:\'IBM Plex Mono\',monospace;padding:0.1rem 0.45rem;border-radius:3px;margin-right:0.3rem;font-weight:500;letter-spacing:0.04em;">{label}</span>'


def difficulty_badge(d: str) -> str:
    color = DIFFICULTY_COLOR.get(d, "#888")
    return f'<span style="background:{color};color:#fff;font-size:0.65rem;padding:0.1rem 0.4rem;border-radius:3px;font-weight:600;">{d}</span>'


def week_badge(w) -> str:
    return f'<span style="background:#1a3a5c;color:#fff;font-size:0.65rem;padding:0.1rem 0.4rem;border-radius:3px;font-weight:600;font-family:\'IBM Plex Mono\',monospace;">Wk{w}</span>'


def render_result_card(row: pd.Series, query_terms: list, show_annotations: bool) -> None:
    text_highlighted = highlight_text(row["text"], query_terms) if query_terms else row["text"]
    ann_html = ""
    if show_annotations and row.get("annotations"):
        parts = []
        for ann in row["annotations"]:
            if not isinstance(ann, dict):
                continue
            lbl = ann.get("label", "")
            span = ann.get("span", "")
            desc = LABEL_DESCRIPTIONS.get(lbl, lbl)
            lbadge = label_badge(lbl)
            parts.append(
                f'<div style="margin-bottom:0.4rem;">{lbadge} <span style="font-size:0.72rem;color:#6b6357;">{desc}</span>'
                f'<div class="result-card ann-span">{span}</div></div>'
            )
        if parts:
            ann_html = (
                '<div class="ann-block" style="margin-top:0.6rem;"><div style="font-size:0.7rem;color:#6b6357;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;margin-bottom:0.4rem;">Annotationen</div>'
                + "".join(parts)
                + "</div>"
            )

    labels_html = " ".join(label_badge(l) for l in sorted(set(row["annotation_labels"]))) if row["annotation_labels"] else ""
    score_html = f'<span class="score-badge">score {row.get("score", 0)}</span>' if row.get("score", 0) > 0 else ""
    wk = week_badge(row["course_week"]) if row.get("course_week") else ""
    diff = difficulty_badge(row["difficulty"]) if row.get("difficulty") else ""
    tt = f'<span class="badge">{row["text_type"]}</span>' if row.get("text_type") else ""

    card_html = f"""
<div class="result-card">
  <div class="meta">
    <span style="font-weight:600;color:#1a3a5c;">{row['doc_id']}</span>
    <span>§ {row['section']} · p{row['paragraph_index']}</span>
    {wk} {diff} {tt} {score_html}
  </div>
  <h4>{row['title']}</h4>
  <div class="body-text">{text_highlighted}</div>
  <div style="margin-top:0.4rem;">{labels_html}</div>
  {ann_html}
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 AkademiKorpus")
    st.markdown(
        '<p style="font-size:0.75rem;opacity:0.7;margin-top:-0.5rem;">Wissenschaftssprache Deutsch</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Korpus laden")
    uploaded = st.file_uploader("JSON hochladen", type=["json"], label_visibility="collapsed")
    path_input = st.text_input("Oder lokaler Pfad", value=str(DEFAULT_JSON_PATH))
    load_clicked = st.button("Korpus laden", type="primary")

    corpus_path = None
    if uploaded is not None:
        tmp = Path("/tmp/uploaded_corpus.json")
        tmp.write_bytes(uploaded.getvalue())
        corpus_path = str(tmp)
    elif load_clicked or Path(path_input).exists():
        corpus_path = path_input

    st.divider()

    if corpus_path and Path(corpus_path).exists():
        try:
            _df_sidebar = corpus_to_dataframe(corpus_path)
            _data_sidebar = load_corpus(corpus_path)

            st.markdown("### 🔍 Suche & Filter")

            query = st.text_input(
                "Stichwortsuche",
                placeholder="z.B. Konjunktiv, Passiv, These, Fragestellung",
            )

            # Week filter
            weeks_available = sorted(_df_sidebar["course_week"].dropna().unique().astype(int).tolist())
            week_options = {f"Woche {w} – {WEEK_TOPICS.get(w, '?')}": w for w in weeks_available}
            selected_week_labels = st.multiselect(
                "Kurswoche",
                options=list(week_options.keys()),
                default=[],
            )
            selected_weeks = [week_options[l] for l in selected_week_labels]

            # Doc filter
            selected_docs = st.multiselect(
                "Dokument",
                options=sorted(_df_sidebar["doc_id"].dropna().unique().tolist()),
                default=[],
            )

            # Section filter
            selected_sections = st.multiselect(
                "Textabschnitt",
                options=sorted(_df_sidebar["section"].dropna().unique().tolist()),
                default=[],
            )

            # Difficulty filter
            diff_options = sorted(_df_sidebar["difficulty"].dropna().unique().tolist())
            selected_difficulty = st.multiselect(
                "Schwierigkeitsgrad",
                options=diff_options,
                default=[],
            )

            # Annotation label filter
            all_labels_available = sorted(
                {label for labels in _df_sidebar["annotation_labels"] for label in labels if label}
            )
            label_display = {LABEL_DESCRIPTIONS.get(l, l): l for l in all_labels_available}
            selected_label_display = st.multiselect(
                "Annotation",
                options=list(label_display.keys()),
                default=[],
            )
            selected_labels = [label_display[l] for l in selected_label_display]

            st.divider()
            st.markdown("### ⚙️ Anzeige")
            max_results = st.slider("Max. Treffer", 5, 80, 20, step=5)
            show_annotations = st.toggle("Annotationen anzeigen", value=True)

        except Exception:
            query = ""
            selected_docs = []
            selected_sections = []
            selected_labels = []
            selected_weeks = []
            selected_difficulty = []
            max_results = 20
            show_annotations = True
    else:
        query = ""
        selected_docs = []
        selected_sections = []
        selected_labels = []
        selected_weeks = []
        selected_difficulty = []
        max_results = 20
        show_annotations = True

# ── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="corpus-header">
  <h1>AkademiKorpus</h1>
  <p>Lehrkorpus für Wissenschaftssprache Deutsch · STBA YAPARI ABA · Sommersemester 2025/2026</p>
</div>
""",
    unsafe_allow_html=True,
)

if corpus_path is None or not Path(corpus_path).exists():
    st.markdown(
        '<div class="empty-state">Bitte eine JSON-Datei hochladen oder einen lokalen Pfad angeben, um den Korpus zu laden.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    df = corpus_to_dataframe(corpus_path)
    data = load_corpus(corpus_path)
except Exception as e:
    st.error(f"Korpus konnte nicht geladen werden: {e}")
    st.stop()

documents = data.get("documents", [])

# ── Metrics ──────────────────────────────────────────────────────────────────
total_paras = len(df)
total_anns = sum(len(r) for r in df["annotations"])
unique_labels = len({l for labels in df["annotation_labels"] for l in labels if l})
weeks_covered = df["course_week"].nunique()

st.markdown(
    f"""
<div class="metric-row">
  <div class="metric-card"><div class="val">{len(documents)}</div><div class="lbl">Dokumente</div></div>
  <div class="metric-card"><div class="val">{total_paras}</div><div class="lbl">Absätze</div></div>
  <div class="metric-card"><div class="val">{total_anns}</div><div class="lbl">Annotationen</div></div>
  <div class="metric-card"><div class="val">{unique_labels}</div><div class="lbl">Label-Typen</div></div>
  <div class="metric-card"><div class="val">{weeks_covered}</div><div class="lbl">Kurswochen</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Filtering ─────────────────────────────────────────────────────────────────
filtered = df.copy()

if selected_weeks:
    filtered = filtered[filtered["course_week"].isin(selected_weeks)]
if selected_docs:
    filtered = filtered[filtered["doc_id"].isin(selected_docs)]
if selected_sections:
    filtered = filtered[filtered["section"].isin(selected_sections)]
if selected_difficulty:
    filtered = filtered[filtered["difficulty"].isin(selected_difficulty)]
if selected_labels:
    filtered = filtered[
        filtered["annotation_labels"].apply(lambda ls: any(l in selected_labels for l in ls))
    ]

if query.strip():
    terms = tokenize_query(query)
    filtered = filtered.copy()
    filtered["score"] = filtered.apply(lambda row: score_row(row, terms), axis=1)
    filtered = filtered[filtered["score"] > 0].sort_values(
        ["score", "doc_id", "section", "paragraph_index"],
        ascending=[False, True, True, True],
    )
else:
    filtered = filtered.copy()
    filtered["score"] = 0
    filtered = filtered.sort_values(["course_week", "doc_id", "section", "paragraph_index"])

query_terms_for_highlight = tokenize_query(query) if query.strip() else []

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_browse, tab_stats, tab_export = st.tabs(
    ["🔎 Suchergebnisse", "📚 Dokument-Browser", "📊 Statistik", "⬇ Export"]
)

# ── Tab 1: Search results ─────────────────────────────────────────────────────
with tab_search:
    display_df = filtered.head(max_results)

    if display_df.empty:
        st.markdown(
            '<div class="empty-state">Keine Treffer für die aktuelle Suche und Filterauswahl.</div>',
            unsafe_allow_html=True,
        )
    else:
        total_hits = len(filtered)
        shown = len(display_df)
        st.markdown(
            f'<p style="font-size:0.8rem;color:#6b6357;margin-bottom:0.75rem;">'
            f'{total_hits} Treffer · {shown} angezeigt</p>',
            unsafe_allow_html=True,
        )

        if selected_labels:
            label_filter_html = "Filter: " + " ".join(label_badge(l) for l in selected_labels)
            st.markdown(f'<p style="margin-bottom:0.75rem;">{label_filter_html}</p>', unsafe_allow_html=True)

        for _, row in display_df.iterrows():
            render_result_card(row, query_terms_for_highlight, show_annotations)

# ── Tab 2: Document browser ───────────────────────────────────────────────────
with tab_browse:
    doc_list = sorted(df["doc_id"].dropna().unique().tolist())
    selected_browse_doc = st.selectbox(
        "Dokument auswählen",
        options=doc_list,
        format_func=lambda d: f"{d} – {df[df['doc_id']==d]['title'].iloc[0]}",
    )

    if selected_browse_doc:
        doc_df = df[df["doc_id"] == selected_browse_doc]
        doc_meta = next((d for d in documents if d["doc_id"] == selected_browse_doc), {})

        title = doc_meta.get("title", "")
        domain = doc_meta.get("domain", "")
        topic = doc_meta.get("topic", "")
        week = doc_meta.get("course_week", "")
        diff = doc_meta.get("difficulty", "")
        ttype = doc_meta.get("text_type", "")

        st.markdown(
            f"""
<div style="background:#f4f1eb;border:1.5px solid #d4c9b8;border-radius:6px;padding:1rem 1.25rem;margin-bottom:1rem;">
  <div style="font-family:'IBM Plex Serif',serif;font-size:1.2rem;font-weight:600;color:#1a3a5c;margin-bottom:0.4rem;">{title}</div>
  <div style="font-size:0.8rem;color:#6b6357;display:flex;gap:0.75rem;flex-wrap:wrap;">
    <span>📚 {domain}</span>
    <span>🏷 {topic}</span>
    <span>{week_badge(week)}</span>
    <span>{difficulty_badge(diff)}</span>
    <span style="background:#e8e3da;padding:0.1rem 0.4rem;border-radius:3px;font-size:0.72rem;">{ttype}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        sections_in_doc = doc_df["section"].unique().tolist()
        for sec in sections_in_doc:
            sec_df = doc_df[doc_df["section"] == sec]
            st.markdown(
                f'<div style="font-family:\'IBM Plex Sans\',sans-serif;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#6b6357;margin-top:1rem;margin-bottom:0.5rem;">{sec}</div>',
                unsafe_allow_html=True,
            )
            for _, row in sec_df.iterrows():
                render_result_card(row, [], show_annotations)

# ── Tab 3: Statistics ─────────────────────────────────────────────────────────
with tab_stats:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Annotationen nach Label**")
        all_ann_labels = [l for ls in df["annotation_labels"] for l in ls if l]
        label_counts = Counter(all_ann_labels)
        label_df = pd.DataFrame(
            [
                {
                    "Label": LABEL_DESCRIPTIONS.get(k, k),
                    "Code": k,
                    "Anzahl": v,
                }
                for k, v in sorted(label_counts.items(), key=lambda x: -x[1])
            ]
        )
        st.dataframe(label_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Dokumente nach Kurswoche**")
        week_doc_df = (
            df.drop_duplicates("doc_id")[["course_week", "doc_id", "title", "difficulty"]]
            .sort_values("course_week")
        )
        week_doc_df.columns = ["Woche", "ID", "Titel", "Schwierigkeit"]
        st.dataframe(week_doc_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Annotationsverteilung nach Textabschnitt**")
    section_label_rows = []
    for _, row in df.iterrows():
        for lbl in row["annotation_labels"]:
            section_label_rows.append({"section": row["section"], "label": lbl})
    if section_label_rows:
        sec_lbl_df = pd.DataFrame(section_label_rows)
        pivot = sec_lbl_df.groupby(["section", "label"]).size().unstack(fill_value=0)
        st.dataframe(pivot, use_container_width=True)

# ── Tab 4: Export ─────────────────────────────────────────────────────────────
with tab_export:
    st.markdown("Gefilterte Ergebnisse exportieren:")
    export_df = filtered.drop(columns=["search_blob"], errors="ignore")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ CSV herunterladen",
            data=csv_bytes,
            file_name="korpus_ergebnisse.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_b:
        jsonl_text = export_df.to_json(orient="records", lines=True, force_ascii=False)
        st.download_button(
            "⬇ JSONL herunterladen",
            data=jsonl_text.encode("utf-8"),
            file_name="korpus_ergebnisse.jsonl",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        # Export just annotations as a flat table
        ann_rows = []
        for _, row in export_df.iterrows():
            for ann in row.get("annotations", []):
                if isinstance(ann, dict):
                    ann_rows.append({
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "section": row["section"],
                        "paragraph_index": row["paragraph_index"],
                        "label": ann.get("label", ""),
                        "span": ann.get("span", ""),
                    })
        ann_export_df = pd.DataFrame(ann_rows)
        if not ann_export_df.empty:
            ann_csv = ann_export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Nur Annotationen (CSV)",
                data=ann_csv,
                file_name="annotationen.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.divider()
    with st.expander("Rohdaten (gefiltert)"):
        st.dataframe(export_df, use_container_width=True, hide_index=True)
