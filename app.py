import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Corpus Query System – Wissenschaftssprache Deutsch",
    page_icon="📚",
    layout="wide",
)

DEFAULT_JSON_PATH = Path("german_academic_corpus_rich_v4.json")

SECTION_ORDER = [
    "Abstract",
    "Einleitung",
    "Theoretischer Hintergrund",
    "Stand der Forschung",
    "Methode",
    "Methodik",
    "Ergebnisse",
    "Analyse",
    "Diskussion",
    "Fazit",
    "Schluss",
]

ANNOTATION_ORDER = [
    "research_question",
    "problem_statement",
    "definition",
    "citation",
    "paraphrase",
    "argument",
    "contrast",
    "comparison",
    "data_description",
    "method_description",
    "result_statement",
    "limitation",
    "passive_style",
]


@st.cache_data(show_spinner=False)
def load_corpus(json_path: str) -> dict:
    """Load the nested corpus JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def corpus_to_dataframe(json_path: str) -> pd.DataFrame:
    """Flatten the nested corpus into one row per paragraph."""
    data = load_corpus(json_path)
    documents = data.get("documents", [])

    rows = []
    for doc in documents:
        base = {
            "doc_id": doc.get("doc_id", ""),
            "title": doc.get("title", ""),
            "domain": doc.get("domain", ""),
            "topic": doc.get("topic", ""),
            "course_target": doc.get("course_target", ""),
        }

        for section_obj in doc.get("sections", []):
            section_name = section_obj.get("section", "")
            for para_idx, para in enumerate(section_obj.get("paragraphs", []), start=1):
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


def tokenize_query(query: str):
    return [t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if t]


def score_row(row: pd.Series, query_terms: list[str]) -> int:
    blob = row.get("search_blob", "")
    if not isinstance(blob, str):
        return 0
    return sum(blob.count(term) for term in query_terms)


def filter_by_annotations(df: pd.DataFrame, selected_labels: list[str]) -> pd.DataFrame:
    if not selected_labels:
        return df
    return df[df["annotation_labels"].apply(lambda labels: any(label in selected_labels for label in labels))]


def normalize_section_name(section: str) -> str:
    section = (section or "").strip()
    if section in {"Stand der Forschung", "Theoretischer Hintergrund"}:
        return "Theoretischer Hintergrund / Stand der Forschung"
    if section in {"Methode", "Methodik"}:
        return "Methode / Methodik"
    if section in {"Ergebnisse", "Analyse"}:
        return "Ergebnisse / Analyse"
    if section in {"Fazit", "Schluss"}:
        return "Fazit / Schluss"
    return section


def render_annotations(annotations: list[dict]) -> str:
    if not annotations:
        return "—"
    parts = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        label = ann.get("label", "")
        span = ann.get("span", "")
        if label and span:
            parts.append(f"**{label}**: {span}")
        elif label:
            parts.append(f"**{label}**")
    return "<br>".join(parts) if parts else "—"


def main():
    st.title("Corpus Query System – Wissenschaftssprache Deutsch")
    st.caption("MVP for searching authentic German academic sections with paragraph-level annotations.")

    with st.sidebar:
        st.header("Corpus")
        uploaded = st.file_uploader("Upload JSON corpus", type=["json"])
        path_input = st.text_input(
            "Or use local file path",
            value=str(DEFAULT_JSON_PATH),
            help="Path to the corpus JSON file on the server running Streamlit.",
        )

        load_clicked = st.button("Load corpus", type="primary")

    corpus_path = None
    if uploaded is not None:
        tmp_path = Path("/tmp/uploaded_corpus.json")
        tmp_path.write_bytes(uploaded.getvalue())
        corpus_path = str(tmp_path)
    elif load_clicked or Path(path_input).exists():
        corpus_path = path_input

    if corpus_path is None or not Path(corpus_path).exists():
        st.info("Upload a JSON file or point to a valid local path to start.")
        st.stop()

    try:
        df = corpus_to_dataframe(corpus_path)
        data = load_corpus(corpus_path)
    except Exception as e:
        st.error(f"Could not load corpus: {e}")
        st.stop()

    documents = data.get("documents", [])
    total_docs = len(documents)
    total_sections = df["section"].nunique() if not df.empty else 0
    total_paragraphs = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", total_docs)
    c2.metric("Sections", total_sections)
    c3.metric("Paragraphs", total_paragraphs)
    c4.metric("Annotation types", df["annotation_label_text"].nunique() if not df.empty else 0)

    st.divider()

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Search")
        query = st.text_input("Keyword query", placeholder="e.g. Korpus, Methode, Vergleich, Datensatz")
        selected_docs = st.multiselect(
            "Document",
            options=sorted(df["doc_id"].dropna().unique().tolist()),
            default=[],
        )
        selected_sections = st.multiselect(
            "Section",
            options=sorted({normalize_section_name(s) for s in df["section"].dropna().unique()}),
            default=[],
        )
        selected_labels = st.multiselect(
            "Annotation labels",
            options=sorted({label for labels in df["annotation_labels"] for label in labels}),
            default=[],
        )
        min_len = st.slider("Minimum paragraph length", 0, int(df["text"].str.len().max() or 0), 0)
        max_results = st.slider("Max results", 5, 100, 20)

        st.subheader("Browse")
        browse_mode = st.radio(
            "View mode",
            ["Search results", "Grouped by document", "Section overview", "Raw dataframe"],
            index=0,
        )

    filtered = df.copy()

    if selected_docs:
        filtered = filtered[filtered["doc_id"].isin(selected_docs)]

    if selected_sections:
        filtered = filtered[filtered["section"].apply(lambda s: normalize_section_name(s) in selected_sections)]

    if selected_labels:
        filtered = filter_by_annotations(filtered, selected_labels)

    filtered = filtered[filtered["text"].fillna("").str.len() >= min_len]

    if query.strip():
        terms = tokenize_query(query)
        filtered = filtered.copy()
        filtered["score"] = filtered.apply(lambda row: score_row(row, terms), axis=1)
        filtered = filtered[filtered["score"] > 0]
        filtered = filtered.sort_values(["score", "doc_id", "section", "paragraph_index"], ascending=[False, True, True, True])
    else:
        filtered = filtered.copy()
        filtered["score"] = 0
        filtered = filtered.sort_values(["doc_id", "section", "paragraph_index"])

    if len(filtered) > max_results:
        filtered = filtered.head(max_results)

    with right:
        if browse_mode == "Search results":
            st.subheader("Matched paragraphs")
            if filtered.empty:
                st.warning("No matches found.")
            else:
                for _, row in filtered.iterrows():
                    with st.expander(f"{row['doc_id']} · {row['section']} · p{row['paragraph_index']} · score {row['score']}"):
                        st.markdown(f"**Title:** {row['title']}")
                        st.markdown(f"**Topic:** {row['topic']}")
                        st.markdown(f"**Section:** {row['section']}")
                        st.markdown(f"**Text:**\n\n{row['text']}")
                        if row.get("annotated_text"):
                            st.markdown(f"**Annotated text:**\n\n{row['annotated_text']}")
                        st.markdown("**Annotations:**", help="Span-level labels inside the paragraph")
                        st.markdown(render_annotations(row["annotations"]), unsafe_allow_html=True)

        elif browse_mode == "Grouped by document":
            st.subheader("Grouped view")
            if filtered.empty:
                st.warning("No data after filtering.")
            else:
                for doc_id, g in filtered.groupby("doc_id", sort=True):
                    doc_title = g["title"].iloc[0]
                    st.markdown(f"### {doc_id} — {doc_title}")
                    st.write(g[["section", "paragraph_index", "score", "annotation_label_text"]])

        elif browse_mode == "Section overview":
            st.subheader("Section overview")
            if filtered.empty:
                st.warning("No data after filtering.")
            else:
                overview = (
                    filtered.groupby(["section", "doc_id"], as_index=False)
                    .agg(paragraphs=("paragraph_index", "count"), annotations=("annotation_label_text", lambda s: ", ".join(sorted(set(x for x in s if x)))))
                )
                st.dataframe(overview, use_container_width=True, hide_index=True)

        else:
            st.subheader("Raw dataframe")
            st.dataframe(filtered, use_container_width=True, hide_index=True)

    with st.expander("Corpus structure preview"):
        if documents:
            first = documents[0]
            st.json(
                {
                    "doc_id": first.get("doc_id"),
                    "title": first.get("title"),
                    "sections": [s.get("section") for s in first.get("sections", [])],
                }
            )

    st.divider()
    st.subheader("Export filtered results")

    export_format = st.radio("Export format", ["CSV", "JSONL"], horizontal=True)
    export_df = filtered.drop(columns=["search_blob"], errors="ignore")

    if export_format == "CSV":
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="corpus_results.csv",
            mime="text/csv",
        )
    else:
        jsonl_text = export_df.to_json(orient="records", lines=True, force_ascii=False)
        st.download_button(
            "Download JSONL",
            data=jsonl_text.encode("utf-8"),
            file_name="corpus_results.jsonl",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
