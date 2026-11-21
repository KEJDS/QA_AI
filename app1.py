import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        PdfReader = None

try:
    import docx
except ImportError:
    docx = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="BugTriage-NLP",
    page_icon="🐞",
    layout="wide",
)

MODEL_NAME = "gemini-3.8-flash"
DUPLICATE_THRESHOLD = 0.85

ABSURD_PATTERNS = [
    r"\b(flat earth|aliens?|time travel|teleport|unicorn|dragon|magic spell|zombie)\b",
    r"\b(gravity reversed|physics broke|universe exploded)\b",
    r"\b(lol+ right now|just kidding|jk+|april fools)\b",
]


# ==============================================================================
# LOCAL PHASE 1 CLASSIFIER, ABSURDITY FILTER & DUPLICATE DETECTOR
# ==============================================================================
def check_absurdity(text: str) -> Tuple[bool, str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return False, ""
    for pat in ABSURD_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            return True, f"Matched non-technical / absurd pattern: {pat}"
    return False, ""


@st.cache_resource
def train_local_classifier() -> Tuple[TfidfVectorizer, LogisticRegression]:
    samples: List[Tuple[str, str]] = [
        (
            "Steps to reproduce: 1. Go to /login. 2. Enter valid email and wrong password. "
            "3. Click Login. Expected result: Show invalid credentials message. "
            "Actual result: Returns HTTP 500 Internal Server Error.",
            "Valid",
        ),
        (
            "On checkout page, applying discount code SAVE10 fails to update total. "
            "Expected: Subtotal decreases by 10%. Actual: Cart total remains unchanged.",
            "Valid",
        ),
        (
            "Exporting CSV with over 5000 rows from Reports page times out after 30s. "
            "Expected: CSV file downloads cleanly. Actual: 504 Gateway Timeout occurs.",
            "Valid",
        ),
        (
            "In Profile settings, uploading a PNG avatar and refreshing reverts to default avatar. "
            "Expected: Uploaded image persists. Actual: Avatar resets on page reload.",
            "Valid",
        ),
        (
            "Filtering orders by start date without end date throws SQL syntax error. "
            "Expected: Display matching orders from start date. Actual: HTTP 400 error.",
            "Valid",
        ),
        ("Login broken", "Missing_Details"),
        ("Bug on dashboard", "Missing_Details"),
        ("Button does not work", "Missing_Details"),
        ("App crashes sometimes", "Missing_Details"),
        ("Wrong output shown", "Missing_Details"),
    ]

    texts = [x[0] for x in samples]
    labels = [x[1] for x in samples]

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    x_train = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_train, labels)
    return vec, clf


def validate_structure_locally(
    report_text: str,
    vec: TfidfVectorizer,
    clf: LogisticRegression,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    clean = (report_text or "").strip()

    absurd, absurd_reason = check_absurdity(clean)
    if not clean:
        return {
            "status": "Missing_Details",
            "confidence": 0.0,
            "is_absurd": False,
            "absurd_reason": "",
            "duplicate": None,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    probs = clf.predict_proba(vec.transform([clean]))[0]
    classes = list(clf.classes_)
    valid_idx = classes.index("Valid") if "Valid" in classes else 1
    valid_prob = float(probs[valid_idx])

    has_details = len(clean.split()) >= 10 and any(
        w in clean.lower() for w in ["expected", "actual", "step", "error", "fail", "timeout"]
    )
    is_valid = (valid_prob >= 0.55) and has_details and (not absurd)
    status = "Valid" if is_valid else "Missing_Details"

    duplicate_info = None
    past_texts = [item.get("raw_report", "") for item in history if item.get("raw_report")]
    if past_texts:
        dup_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(
            past_texts + [clean]
        )
        sims = cosine_similarity(dup_vec[-1], dup_vec[:-1])[0]
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        if best_sim >= DUPLICATE_THRESHOLD:
            matched = history[best_i]
            duplicate_info = {
                "similarity": round(best_sim * 100, 1),
                "title": matched.get("ticket_title", "Previous Defect"),
                "component": matched.get("component", "General"),
            }

    return {
        "status": status,
        "confidence": round(valid_prob * 100, 2),
        "is_absurd": absurd,
        "absurd_reason": absurd_reason,
        "duplicate": duplicate_info,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ==============================================================================
# DOCUMENT PARSING, GEMINI & GITHUB EXPORT
# ==============================================================================
def parse_uploaded_file(uploaded_file) -> Tuple[str, Optional[str]]:
    if uploaded_file is None:
        return "", "No file uploaded."

    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    try:
        if name.endswith(".txt"):
            return data.decode("utf-8", errors="replace"), None
        if name.endswith(".pdf"):
            if PdfReader is None:
                return "", "Install pypdf or PyPDF2 to read PDF files."
            reader = PdfReader(io.BytesIO(data))
            return "\n".join([(p.extract_text() or "") for p in reader.pages]).strip(), None
        if name.endswith(".docx"):
            if docx is None:
                return "", "Install python-docx to read DOCX files."
            doc = docx.Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip(), None
        return "", "Supported formats: .txt, .pdf, .docx"
    except Exception as exc:
        return "", f"Error reading file: {exc}"


def init_gemini() -> Tuple[Optional[Any], Optional[str]]:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")  # type: ignore
    if not api_key:
        return None, "GEMINI_API_KEY not configured."
    if genai is None:
        return None, "google-genai SDK not installed."
    try:
        return genai.Client(api_key=api_key), None
    except Exception as exc:
        return None, f"Failed to initialize Gemini client: {exc}"


def refine_and_cross_reference_prd(
    client: Any,
    raw_report: str,
    prd_text: str,
    guidelines_text: str,
) -> Tuple[Dict[str, Any], float, bool]:
    t0 = time.perf_counter()

    prompt = f"""
You are a QA triage assistant. Review the bug report against the uploaded PRD and QA guidelines.
Return ONLY valid JSON with these keys:
- "ticket_title": concise bug title
- "severity": "Critical" | "High" | "Medium" | "Low"
- "component": target software component
- "coherency_status": "Coherent" | "Contradictory / Incoherent"
- "prd_alignment": explanation of how the report matches or conflicts with the PRD
- "structured_report": markdown formatted defect report

PRD:
\"\"\"{prd_text[:9000]}\"\"\"

GUIDELINES:
\"\"\"{guidelines_text[:3000]}\"\"\"

RAW BUG REPORT:
\"\"\"{raw_report}\"\"\"
"""

    if client is not None:
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            raw_out = (getattr(resp, "text", "") or "").strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_out, flags=re.DOTALL).strip()
            parsed = json.loads(cleaned)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return parsed, elapsed, False
        except Exception:
            pass

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    summary_line = (raw_report or "Defect").strip().splitlines()[0][:80]
    fallback = {
        "ticket_title": f"[Bug] {summary_line}",
        "severity": "Medium",
        "component": "General",
        "coherency_status": "Local Fallback",
        "prd_alignment": "Gemini unavailable or quota reached; fell back to local processing.",
        "structured_report": (
            f"### {summary_line}\n"
            f"- **Severity:** Medium\n"
            f"- **Component:** General\n"
            f"- **Details:** {raw_report}"
        ),
    }
    return fallback, elapsed, True


def export_to_github_issue(
    token: str,
    repo_full_name: str,
    title: str,
    body: str,
) -> Tuple[bool, str]:
    if not token or not repo_full_name:
        return False, "Provide both a GitHub token and `owner/repo`."
    url = f"https://api.github.com/repos/{repo_full_name.strip()}/issues"
    headers = {
        "Authorization": f"Bearer {token.strip()}",
        "Accept": "application/vnd.github+json",
    }
    try:
        r = requests.post(url, headers=headers, json={"title": title, "body": body}, timeout=15)
        if r.status_code in (200, 201):
            issue_url = r.json().get("html_url", "")
            return True, f"Created GitHub issue: {issue_url}"
        return False, f"GitHub API error ({r.status_code}): {r.text[:200]}"
    except Exception as exc:
        return False, f"GitHub request failed: {exc}"


# ==============================================================================
# MONGODB LOGGING
# ==============================================================================
@st.cache_resource
def get_mongo_collection():
    uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI", "")  # type: ignore
    db_name = os.getenv("MONGO_DB", "bugtriage_nlp")
    col_name = os.getenv("MONGO_COLLECTION", "triage_logs")
    if not uri or MongoClient is None:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client[db_name][col_name]
    except Exception:
        return None


def log_record(entry: Dict[str, Any]) -> None:
    doc = dict(entry)
    doc["timestamp"] = datetime.now(timezone.utc).isoformat()
    st.session_state["records"].append(doc)
    col = get_mongo_collection()
    if col is not None:
        try:
            col.insert_one(dict(doc))
        except Exception:
            pass


def fetch_records() -> List[Dict[str, Any]]:
    col = get_mongo_collection()
    if col is not None:
        try:
            rows = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(200))
            if rows:
                return rows
        except Exception:
            pass
    return list(reversed(st.session_state.get("records", [])))


# ==============================================================================
# SESSION STATE
# ==============================================================================
for key, default in {
    "prd_text": "",
    "prd_name": "",
    "guidelines_text": "",
    "phase1_out": None,
    "phase2_out": None,
    "records": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==============================================================================
# UI
# ==============================================================================
st.title("🐞 BugTriage-NLP")
st.caption(f"Hybrid Defect Validation & Triage • Local ML + {MODEL_NAME} + MongoDB + GitHub")

vec, clf = train_local_classifier()
gemini_client, gemini_err = init_gemini()
records = fetch_records()

tab1, tab2 = st.tabs(["Triage Workspace", "QA Dashboard"])

with tab1:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("PRD Upload (Gatekeeper)")
        prd_upload = st.file_uploader(
            "Upload PRD (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            key="prd_file",
        )
        if prd_upload is not None:
            text_val, err_val = parse_uploaded_file(prd_upload)
            if err_val:
                st.error(err_val)
            else:
                st.session_state["prd_text"] = text_val
                st.session_state["prd_name"] = prd_upload.name
                st.success(f"PRD loaded: {prd_upload.name}")

        guide_upload = st.file_uploader(
            "Optional QA Guidelines (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            key="guide_file",
        )
        if guide_upload is not None:
            g_text, g_err = parse_uploaded_file(guide_upload)
            if not g_err:
                st.session_state["guidelines_text"] = g_text

        st.subheader("Defect Report Input")
        raw_bug = st.text_area(
            "Raw defect report",
            height=180,
            placeholder="Describe reproduction steps, expected result, and actual result...",
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Validate Structure (Phase 1)", use_container_width=True):
                st.session_state["phase1_out"] = validate_structure_locally(
                    raw_bug, vec, clf, records
                )

        with col_b2:
            prd_ready = bool(st.session_state["prd_text"].strip())
            if st.button(
                "Triage with PRD & AI (Phase 2)",
                type="primary",
                disabled=not prd_ready,
                use_container_width=True,
            ):
                p1 = validate_structure_locally(raw_bug, vec, clf, records)
                st.session_state["phase1_out"] = p1
                p2, p2_ms, used_fallback = refine_and_cross_reference_prd(
                    gemini_client,
                    raw_bug,
                    st.session_state["prd_text"],
                    st.session_state["guidelines_text"],
                )
                st.session_state["phase2_out"] = {
                    **p2,
                    "phase2_ms": p2_ms,
                    "fallback": used_fallback,
                }
                log_record(
                    {
                        "raw_report": raw_bug,
                        "prd_name": st.session_state["prd_name"],
                        "phase1_status": p1["status"],
                        "phase1_ms": p1["elapsed_ms"],
                        "phase2_ms": p2_ms,
                        "fallback": used_fallback,
                        "ticket_title": p2.get("ticket_title", ""),
                        "severity": p2.get("severity", "Medium"),
                        "component": p2.get("component", "General"),
                    }
                )
            if not prd_ready:
                st.caption("Upload a PRD document first to unlock Phase 2 triage.")

    with right:
        st.subheader("Validation & Triage Output")
        if gemini_err:
            st.caption(gemini_err)

        p1_res = st.session_state.get("phase1_out")
        if p1_res:
            if p1_res["is_absurd"]:
                st.error(f"Absurd input detected: {p1_res['absurd_reason']}")
            elif p1_res["status"] == "Valid":
                st.success(
                    f"Phase 1: Valid structure ({p1_res['confidence']}% • {p1_res['elapsed_ms']} ms)"
                )
            else:
                st.warning(
                    f"Phase 1: Missing Details ({p1_res['confidence']}% • {p1_res['elapsed_ms']} ms) — Revise & Resubmit."
                )

            if p1_res.get("duplicate"):
                dup = p1_res["duplicate"]
                st.warning(
                    f"Potential duplicate ({dup['similarity']}% match): {dup['title']} ({dup['component']})"
                )

        p2_res = st.session_state.get("phase2_out")
        if p2_res:
            st.markdown(f"**Title:** {p2_res.get('ticket_title', '')}")
            st.markdown(
                f"**Severity:** {p2_res.get('severity', 'Medium')} | "
                f"**Component:** {p2_res.get('component', 'General')} | "
                f"**Coherency:** {p2_res.get('coherency_status', 'N/A')}"
            )
            st.markdown(f"**PRD Alignment:** {p2_res.get('prd_alignment', '')}")
            st.markdown(p2_res.get("structured_report", ""))

            with st.expander("Export to GitHub Issue"):
                gh_repo = st.text_input("Repository (owner/repo)", key="gh_repo")
                gh_token = st.text_input("GitHub Token", type="password", key="gh_token")
                if st.button("Create GitHub Issue"):
                    ok, msg = export_to_github_issue(
                        gh_token,
                        gh_repo,
                        p2_res.get("ticket_title", "Bug Report"),
                        p2_res.get("structured_report", raw_bug),
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

with tab2:
    st.subheader("QA Analytics")
    all_rows = fetch_records()
    if not all_rows:
        st.info("No triage logs available yet.")
    else:
        df = pd.DataFrame(all_rows)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Logs", len(df))
        m2.metric("Avg Phase 1 (ms)", round(float(df["phase1_ms"].mean()), 2) if "phase1_ms" in df else 0.0)
        m3.metric("Avg Phase 2 (ms)", round(float(df["phase2_ms"].mean()), 2) if "phase2_ms" in df else 0.0)
        st.dataframe(df, use_container_width=True)
