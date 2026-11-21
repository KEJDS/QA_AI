import io
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies for file parsing, Gemini SDK, and MongoDB
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        PdfReader = None

try:
    import docx  # python-docx
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
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="BugTriage-NLP | Hybrid Defect Validation & Triage",
    page_icon="🐞",
    layout="wide",
)

GEMINI_MODEL_NAME = "gemini-3.8-flash"
DUPLICATE_THRESHOLD = 0.85


# ==============================================================================
# 2. LOCAL PHASE 1 CLASSIFIER, ABSURDITY FILTER & DUPLICATE DETECTOR
# ==============================================================================
ABSURDITY_PATTERNS = [
    r"\b(flat earth|aliens?|time travel|teleport|unicorn|dragon|magic spell|zombie)\b",
    r"\b(gravity reversed|physics broke|universe exploded|my coffee brewed itself)\b",
    r"\b(lol+ right now|just kidding|jk+|meme bug|april fools)\b",
    r"\b(1000000% cpu|9999999 gb ram|infinite loop that froze time)\b",
]


def is_absurd_report(text: str) -> Tuple[bool, str]:
    """Local regex-based absurdity/nonsense filter before spending API tokens."""
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False, ""
    for pat in ABSURDITY_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            return True, f"Matched non-technical / absurd pattern: `{pat}`"
    return False, ""


@st.cache_resource
def build_local_classifier() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Builds TF-IDF (unigram + bigram) + Logistic Regression model for Phase 1
    structural completeness classification ("Valid" vs "Missing_Details").
    """
    training_samples: List[Tuple[str, str]] = [
        # Structurally complete / valid defect reports
        (
            "Steps to reproduce: 1. Navigate to /login. 2. Enter valid email and wrong password. "
            "3. Click Sign In. Expected result: Error message 'Invalid credentials' appears. "
            "Actual result: App returns HTTP 500 Internal Server Error on API /auth/login.",
            "Valid",
        ),
        (
            "On checkout page, selecting express shipping and applying promo code SAVE10 fails "
            "to recalculate total. Expected: Discount subtracts 10% from subtotal. "
            "Actual: Total stays unchanged and console logs TypeError in cart.js line 84.",
            "Valid",
        ),
        (
            "When exporting a CSV report with >5000 rows from the Analytics dashboard, "
            "the request times out after 30 seconds. Expected: File downloads within 5 seconds. "
            "Actual: Gateway timeout 504 occurs in backend export endpoint.",
            "Valid",
        ),
        (
            "Repro steps: Open user profile settings, upload a 2MB PNG avatar, click Save. "
            "Expected: Avatar updates and persists on reload. Actual: Profile image reverts "
            "to default placeholder and MongoDB profile document is not updated.",
            "Valid",
        ),
        (
            "Filter by date range on Invoices table throws SQL syntax error when end date is empty. "
            "Expected: Return all invoices from start date onward. Actual: 400 Bad Request "
            "with malformed query parameter error.",
            "Valid",
        ),
        (
            "Mobile view (375px width) overlaps the submit button over the terms checkbox on "
            "registration form. Expected: Checkbox and button stack cleanly without overlap. "
            "Actual: Button blocks tapping the checkbox in Chrome Android.",
            "Valid",
        ),
        # Structurally incomplete / missing details
        (
            "Login broken",
            "Missing_Details",
        ),
        (
            "Bug on dashboard please fix",
            "Missing_Details",
        ),
        (
            "It does not work when I click it",
            "Missing_Details",
        ),
        (
            "Page crashing sometimes",
            "Missing_Details",
        ),
        (
            "Wrong value shown",
            "Missing_Details",
        ),
        (
            "API issue",
            "Missing_Details",
        ),
    ]

    texts = [t for t, _ in training_samples]
    labels = [lbl for _, lbl in training_samples]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    x_vec = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(x_vec, labels)
    return vectorizer, clf


def run_phase1_validation(
    bug_text: str,
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    historical_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Executes Phase 1:
    1. Absurdity check
    2. Structural completeness check (TF-IDF + Logistic Regression + basic heuristics)
    3. Historical duplicate detection via TF-IDF cosine similarity (>= 0.85)
    """
    start_ms = time.perf_counter()
    clean_text = (bug_text or "").strip()

    absurd_flag, absurd_reason = is_absurd_report(clean_text)

    if not clean_text:
        return {
            "status": "Missing_Details",
            "is_valid_structure": False,
            "confidence": 0.0,
            "is_absurd": False,
            "absurd_reason": "",
            "missing_hints": ["Defect description is empty."],
            "duplicate_match": None,
            "elapsed_ms": round((time.perf_counter() - start_ms) * 1000, 2),
        }

    x_input = vectorizer.transform([clean_text])
    probs = clf.predict_proba(x_input)[0]
    classes = list(clf.classes_)
    valid_idx = classes.index("Valid") if "Valid" in classes else 1
    valid_prob = float(probs[valid_idx])

    # Structural heuristics to reinforce short/vague entries
    word_count = len(clean_text.split())
    lower_t = clean_text.lower()
    has_expected_or_actual = any(
        k in lower_t for k in ["expected", "actual", "step", "repro", "error", "fails", "timeout"]
    )

    missing_hints: List[str] = []
    if word_count < 12:
        missing_hints.append("Description is very brief (fewer than 12 words).")
    if not has_expected_or_actual:
        missing_hints.append(
            "Include reproduction steps, expected behavior, and actual behavior/error details."
        )

    is_structurally_valid = (valid_prob >= 0.55) and (word_count >= 10) and (not absurd_flag)
    status_label = "Valid" if is_structurally_valid else "Missing_Details"

    # Historical duplicate check using cosine similarity
    duplicate_match: Optional[Dict[str, Any]] = None
    corpus = [r.get("raw_report", "") for r in historical_reports if r.get("raw_report")]
    if corpus:
        all_docs = corpus + [clean_text]
        dup_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(all_docs)
        sim_scores = cosine_similarity(dup_vec[-1], dup_vec[:-1])[0]
        best_idx = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_idx])
        if best_score >= DUPLICATE_THRESHOLD:
            matched_doc = historical_reports[best_idx]
            duplicate_match = {
                "similarity": round(best_score * 100, 1),
                "matched_title": matched_doc.get("ticket_title", "Historical Defect"),
                "matched_component": matched_doc.get("component", "Unassigned"),
            }

    return {
        "status": status_label,
        "is_valid_structure": is_structurally_valid,
        "confidence": round(valid_prob * 100, 2),
        "is_absurd": absurd_flag,
        "absurd_reason": absurd_reason,
        "missing_hints": missing_hints,
        "duplicate_match": duplicate_match,
        "elapsed_ms": round((time.perf_counter() - start_ms) * 1000, 2),
    }


# ==============================================================================
# 3. DOCUMENT PARSER (.TXT, .PDF, .DOCX) & GEMINI CLIENT
# ==============================================================================
def extract_uploaded_text(uploaded_file) -> Tuple[str, Optional[str]]:
    """Extracts text from .txt, .pdf, or .docx files."""
    if uploaded_file is None:
        return "", "No file provided."

    file_name = (uploaded_file.name or "").lower()
    raw_bytes = uploaded_file.getvalue()

    try:
        if file_name.endswith(".txt"):
            return raw_bytes.decode("utf-8", errors="replace"), None

        if file_name.endswith(".pdf"):
            if PdfReader is None:
                return "", "Install `pypdf` or `PyPDF2` to parse PDF documents."
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_text = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(pages_text).strip(), None

        if file_name.endswith(".docx"):
            if docx is None:
                return "", "Install `python-docx` to parse DOCX documents."
            document = docx.Document(io.BytesIO(raw_bytes))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip(), None

        return "", "Unsupported file type. Please upload a .txt, .pdf, or .docx file."
    except Exception as exc:
        return "", f"Failed to parse uploaded file: {exc}"


def get_gemini_client() -> Tuple[Optional[Any], Optional[str]]:
    """Initializes Google GenAI client if SDK and API key are available."""
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")  # type: ignore
    if not api_key:
        return None, "GEMINI_API_KEY is not set."
    if genai is None:
        return None, "`google-genai` package is not installed."
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as exc:
        return None, f"Could not initialize Gemini client: {exc}"


def call_gemini_with_fallback(
    client: Any,
    prompt: str,
    model_name: str = GEMINI_MODEL_NAME,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Gemini model and returns (response_text, error_message).
    Catches quota/token exhaustion, timeouts, or model errors gracefully.
    """
    if client is None:
        return None, "Gemini client unavailable."
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            return None, "Empty response returned by Gemini."
        return text, None
    except Exception as exc:
        return None, f"Gemini fallback triggered ({type(exc).__name__}): {exc}"


# ==============================================================================
# 4. COMBINED COHERENCY + REQUIREMENTS CHECK & TICKET RESTRUCTURING
# ==============================================================================
def evaluate_coherency_and_requirements(
    client: Any,
    bug_report_text: str,
    prd_text: str,
    guidelines_text: str = "",
) -> Dict[str, Any]:
    """
    Evaluates BOTH:
      1. Internal logical coherency ("coherent")
      2. Alignment with uploaded PRD / instructions ("follows_requirements")
    Returns exact status labels:
      - "Passed Both"
      - "Not Coherent"
      - "Not Accurate to the Instructions"
      - "Not Coherent & Not Accurate to the Instructions"
    With local heuristic fallback if Gemini is unavailable or hits token limits.
    """
    start_ms = time.perf_counter()
    prd_clean = (prd_text or "").strip()
    bug_clean = (bug_report_text or "").strip()
    guide_clean = (guidelines_text or "").strip()

    if not prd_clean:
        return {
            "coherent": None,
            "follows_requirements": False,
            "status_label": "Missing PRD / Requirements Document",
            "coherency_reason": "Cannot verify requirements compliance without an uploaded PRD.",
            "requirements_reason": "Please upload a PRD file (.txt, .pdf, .docx) first.",
            "fallback_used": False,
            "elapsed_ms": round((time.perf_counter() - start_ms) * 1000, 2),
        }

    requirements_context = prd_clean[:10000]
    if guide_clean:
        requirements_context += f"\n\nADDITIONAL QA GUIDELINES:\n{guide_clean[:4000]}"

    prompt = f"""
You are a strict QA & Requirements Verification engine.
Analyze the BUG REPORT against TWO independent checks:

1. COHERENCY ("coherent"):
   - Are the steps, actual behavior, and expected behavior logically consistent?
   - Does it avoid self-contradictions or impossible/nonsensical claims?

2. REQUIREMENTS / INSTRUCTIONS ACCURACY ("follows_requirements"):
   - Does the bug report align with the business rules, expected behaviors, and scope in the UPLOADED PRD / QA INSTRUCTIONS?
   - Mark "follows_requirements": false if the report claims something contradicts the PRD or violates stated instructions/rules.

Respond ONLY with raw JSON (no markdown fences) matching this exact schema:
{{
  "coherent": true or false,
  "follows_requirements": true or false,
  "coherency_reason": "brief explanation",
  "requirements_reason": "brief explanation"
}}

UPLOADED PRD & INSTRUCTIONS:
\"\"\"{requirements_context}\"\"\"

BUG REPORT:
\"\"\"{bug_clean}\"\"\"
"""

    raw_text, err = call_gemini_with_fallback(client, prompt, GEMINI_MODEL_NAME)

    if raw_text and not err:
        try:
            cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
            parsed = json.loads(cleaned_json)
            coherent = bool(parsed.get("coherent", False))
            follows_req = bool(parsed.get("follows_requirements", False))
            coh_reason = str(parsed.get("coherency_reason", "")).strip()
            req_reason = str(parsed.get("requirements_reason", "")).strip()

            if coherent and follows_req:
                status_label = "Passed Both"
            elif (not coherent) and follows_req:
                status_label = "Not Coherent"
            elif coherent and (not follows_req):
                status_label = "Not Accurate to the Instructions"
            else:
                status_label = "Not Coherent & Not Accurate to the Instructions"

            return {
                "coherent": coherent,
                "follows_requirements": follows_req,
                "status_label": status_label,
                "coherency_reason": coh_reason or "Evaluated by Gemini.",
                "requirements_reason": req_reason or "Evaluated against uploaded PRD.",
                "fallback_used": False,
                "elapsed_ms": round((time.perf_counter() - start_ms) * 1000, 2),
            }
        except Exception:
            err = "Malformed JSON returned by model; fell back to local rule check."

    # Graceful local fallback when API is unavailable or token-exhausted
    absurd_flag, absurd_reason = is_absurd_report(bug_clean)
    coherent_local = not absurd_flag

    # Basic keyword overlap check against PRD terms as deterministic fallback
    prd_tokens = {w for w in re.findall(r"[a-z]{4,}", prd_clean.lower())}
    bug_tokens = {w for w in re.findall(r"[a-z]{4,}", bug_clean.lower())}
    overlap_ratio = (len(prd_tokens & bug_tokens) / max(1, len(bug_tokens))) if bug_tokens else 0.0
    follows_req_local = overlap_ratio >= 0.12

    if coherent_local and follows_req_local:
        status_label = "Passed Both"
    elif (not coherent_local) and follows_req_local:
        status_label = "Not Coherent"
    elif coherent_local and (not follows_req_local):
        status_label = "Not Accurate to the Instructions"
    else:
        status_label = "Not Coherent & Not Accurate to the Instructions"

    return {
        "coherent": coherent_local,
        "follows_requirements": follows_req_local,
        "status_label": status_label,
        "coherency_reason": (
            absurd_reason if absurd_flag else "Local fallback: no internal contradiction detected."
        ),
        "requirements_reason": (
            f"Local fallback ({err}): token alignment ratio = {overlap_ratio:.0%} against PRD."
        ),
        "fallback_used": True,
        "elapsed_ms": round((time.perf_counter() - start_ms) * 1000, 2),
    }


def generate_structured_ticket(
    client: Any,
    bug_report_text: str,
    prd_text: str,
    guidelines_text: str,
    check_result: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, bool]:
    """Produces a standardized developer ticket with graceful local fallback."""
    start_ms = time.perf_counter()

    prompt = f"""
Convert the raw bug report into a structured Agile developer ticket using the PRD and QA guidelines.
Return ONLY valid JSON with keys:
- "ticket_title": string
- "severity": one of ["Critical", "High", "Medium", "Low"]
- "component": string (e.g., "Authentication", "Checkout / Billing", "UI / Frontend", "API / Backend", "Reporting")
- "probable_root_cause": string
- "prd_alignment_notes": string
- "formatted_markdown": markdown string with Summary, Steps to Reproduce, Expected vs Actual Behavior, and PRD Notes

PRD:
\"\"\"{prd_text[:8000]}\"\"\"

QA GUIDELINES:
\"\"\"{guidelines_text[:3000]}\"\"\"

COMBINED CHECK STATUS:
{check_result.get("status_label", "Unchecked")} | {check_result.get("coherency_reason", "")} | {check_result.get("requirements_reason", "")}

RAW BUG REPORT:
\"\"\"{bug_report_text}\"\"\"
"""

    raw_text, err = call_gemini_with_fallback(client, prompt, GEMINI_MODEL_NAME)
    elapsed_ms = round((time.perf_counter() - start_ms) * 1000, 2)

    if raw_text and not err:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
            data = json.loads(cleaned)
            return data, elapsed_ms, False
        except Exception:
            pass

    # Fallback ticket structure
    first_line = (bug_report_text or "Defect Report").strip().splitlines()[0][:80]
    fallback_ticket = {
        "ticket_title": f"[Triage] {first_line}",
        "severity": "Medium",
        "component": "General / Unassigned",
        "probable_root_cause": "Generated via local fallback; manual root-cause verification recommended.",
        "prd_alignment_notes": check_result.get("requirements_reason", "Pending PRD review."),
        "formatted_markdown": (
            f"### {first_line}\n"
            f"- **Severity:** Medium\n"
            f"- **Component:** General / Unassigned\n"
            f"- **Verification Status:** {check_result.get('status_label', 'N/A')}\n"
            f"- **Raw Report:** {bug_report_text}\n"
        ),
    }
    return fallback_ticket, elapsed_ms, True


# ==============================================================================
# 5. MONGODB PERSISTENCE & ANALYTICS HELPERS
# ==============================================================================
@st.cache_resource
def get_mongo_collection():
    """Connects to MongoDB collection if MONGO_URI is configured."""
    mongo_uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI", "")  # type: ignore
    db_name = os.getenv("MONGO_DB", "bugtriage_nlp")
    col_name = os.getenv("MONGO_COLLECTION", "triage_logs")
    if not mongo_uri or MongoClient is None:
        return None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client[db_name][col_name]
    except Exception:
        return None


def save_triage_record(record: Dict[str, Any]) -> None:
    """Saves a triage record to MongoDB and session history."""
    record_copy = dict(record)
    record_copy["timestamp"] = datetime.now(timezone.utc).isoformat()
    st.session_state["triage_history"].append(record_copy)

    col = get_mongo_collection()
    if col is not None:
        try:
            col.insert_one(dict(record_copy))
        except Exception:
            pass


def load_triage_records() -> List[Dict[str, Any]]:
    """Loads triage records from MongoDB if available, merged with session history."""
    col = get_mongo_collection()
    records: List[Dict[str, Any]] = []
    if col is not None:
        try:
            cursor = col.find({}, {"_id": 0}).sort("timestamp", -1).limit(200)
            records = list(cursor)
        except Exception:
            records = []
    if not records:
        records = list(reversed(st.session_state.get("triage_history", [])))
    return records


# ==============================================================================
# 6. SESSION STATE INITIALIZATION
# ==============================================================================
if "prd_text" not in st.session_state:
    st.session_state["prd_text"] = ""
if "prd_filename" not in st.session_state:
    st.session_state["prd_filename"] = ""
if "guidelines_text" not in st.session_state:
    st.session_state["guidelines_text"] = ""
if "phase1_result" not in st.session_state:
    st.session_state["phase1_result"] = None
if "combined_check_result" not in st.session_state:
    st.session_state["combined_check_result"] = None
if "generated_ticket" not in st.session_state:
    st.session_state["generated_ticket"] = None
if "triage_history" not in st.session_state:
    st.session_state["triage_history"] = []


# ==============================================================================
# 7. MAIN UI & WORKFLOW
# ==============================================================================
st.title("🐞 BugTriage-NLP")
st.caption(
    f"Hybrid Defect Report Validation & Triage • Phase 1 (TF-IDF + Logistic Regression) "
    f"• Phase 2 ({GEMINI_MODEL_NAME}) • PRD & Coherency Verification"
)

vectorizer, clf = build_local_classifier()
gemini_client, gemini_init_err = get_gemini_client()
existing_records = load_triage_records()

tab_triage, tab_dashboard = st.tabs(["🧪 Defect Triage Workspace", "📊 QA Manager Dashboard"])

with tab_triage:
    col_left, col_right = st.columns([1.05, 1.25], gap="large")

    with col_left:
        st.subheader("1. PRD & QA Formatting Requirements (Required)")
        prd_file = st.file_uploader(
            "Upload Product Requirements Document (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            key="prd_uploader",
        )
        if prd_file is not None:
            parsed_prd, parse_err = extract_uploaded_text(prd_file)
            if parse_err:
                st.error(parse_err)
            else:
                st.session_state["prd_text"] = parsed_prd
                st.session_state["prd_filename"] = prd_file.name
                st.success(f"Loaded PRD: **{prd_file.name}** ({len(parsed_prd)} chars)")

        guide_file = st.file_uploader(
            "Optional Custom QA Formatting Guidelines (.txt, .pdf, .docx)",
            type=["txt", "pdf", "docx"],
            key="guidelines_uploader",
        )
        if guide_file is not None:
            parsed_guide, g_err = extract_uploaded_text(guide_file)
            if g_err:
                st.warning(g_err)
            else:
                st.session_state["guidelines_text"] = parsed_guide
                st.info(f"Loaded Guidelines: **{guide_file.name}**")

        with st.expander("Preview Loaded PRD Text", expanded=False):
            st.text_area(
                "PRD Content",
                value=st.session_state["prd_text"],
                height=160,
                disabled=True,
                label_visibility="collapsed",
            )

        st.divider()
        st.subheader("2. Raw Defect Report Input")
        bug_input = st.text_area(
            "Enter or paste the raw defect report",
            height=190,
            placeholder=(
                "Example: Steps to reproduce: 1. Open Checkout. 2. Apply expired coupon. "
                "Expected: Reject coupon with error message. Actual: Discount applied anyway."
            ),
            key="bug_report_input",
        )

        btn_p1, btn_combined = st.columns(2)

        with btn_p1:
            run_p1_clicked = st.button(
                "Run Phase 1 Check",
                key="btn_phase1",
                use_container_width=True,
            )

        with btn_combined:
            run_combined_clicked = st.button(
                "Check Coherency & Requirements",
                key="btn_check_coherency_and_req",
                type="primary",
                use_container_width=True,
            )

        if run_p1_clicked:
            p1_res = run_phase1_validation(
                bug_text=bug_input,
                vectorizer=vectorizer,
                clf=clf,
                historical_reports=existing_records,
            )
            st.session_state["phase1_result"] = p1_res

        if run_combined_clicked:
            p1_res = run_phase1_validation(
                bug_text=bug_input,
                vectorizer=vectorizer,
                clf=clf,
                historical_reports=existing_records,
            )
            st.session_state["phase1_result"] = p1_res

            comb_res = evaluate_coherency_and_requirements(
                client=gemini_client,
                bug_report_text=bug_input,
                prd_text=st.session_state.get("prd_text", ""),
                guidelines_text=st.session_state.get("guidelines_text", ""),
            )
            st.session_state["combined_check_result"] = comb_res

            ticket_data, p2_ms, p2_fallback = generate_structured_ticket(
                client=gemini_client,
                bug_report_text=bug_input,
                prd_text=st.session_state.get("prd_text", ""),
                guidelines_text=st.session_state.get("guidelines_text", ""),
                check_result=comb_res,
            )
            st.session_state["generated_ticket"] = ticket_data

            save_triage_record(
                {
                    "raw_report": bug_input,
                    "prd_filename": st.session_state.get("prd_filename", "None"),
                    "phase1_status": p1_res["status"],
                    "phase1_confidence": p1_res["confidence"],
                    "phase1_ms": p1_res["elapsed_ms"],
                    "combined_status": comb_res["status_label"],
                    "coherent": comb_res["coherent"],
                    "follows_requirements": comb_res["follows_requirements"],
                    "combined_ms": comb_res["elapsed_ms"],
                    "phase2_ms": p2_ms,
                    "fallback_used": comb_res["fallback_used"] or p2_fallback,
                    "ticket_title": ticket_data.get("ticket_title", ""),
                    "severity": ticket_data.get("severity", "Medium"),
                    "component": ticket_data.get("component", "General"),
                }
            )

    with col_right:
        st.subheader("3. Validation, Coherency & Requirements Results")

        if gemini_init_err:
            st.caption(f"ℹ️ {gemini_init_err} — Local fallback mode active when needed.")

        p1 = st.session_state.get("phase1_result")
        if p1:
            st.markdown("**Phase 1 — Local Structural & Gatekeeper Check**")
            if p1["is_absurd"]:
                st.error(f"🚫 **Absurd / Non-Technical Claim Detected:** {p1['absurd_reason']}")
            elif p1["status"] == "Valid":
                st.success(
                    f"✅ **Structure: Valid** (Confidence: {p1['confidence']}% • {p1['elapsed_ms']} ms)"
                )
            else:
                st.warning(
                    f"⚠️ **Structure: Missing_Details** (Confidence: {p1['confidence']}% • {p1['elapsed_ms']} ms)"
                )
                for hint in p1.get("missing_hints", []):
                    st.write(f"- {hint}")

            if p1.get("duplicate_match"):
                dup = p1["duplicate_match"]
                st.warning(
                    f"🔁 **Possible Historical Duplicate ({dup['similarity']}% similarity):** "
                    f"*{dup['matched_title']}* [{dup['matched_component']}]"
                )

        comb = st.session_state.get("combined_check_result")
        if comb:
            st.markdown("**Phase 2 — Combined Coherency & Uploaded PRD Requirements Check**")
            label = comb["status_label"]
            detail_line = (
                f"**Coherency:** {comb['coherency_reason']}  \n"
                f"**PRD / Requirements:** {comb['requirements_reason']}  \n"
                f"*Latency: {comb['elapsed_ms']} ms"
                f"{' • Fallback Mode' if comb.get('fallback_used') else ''}*"
            )

            if label == "Passed Both":
                st.success(f"✅ **Passed Both**\n\n{detail_line}")
            elif label == "Not Coherent":
                st.error(f"❌ **Not Coherent**\n\n{detail_line}")
            elif label == "Not Accurate to the Instructions":
                st.warning(f"⚠️ **Not Accurate to the Instructions**\n\n{detail_line}")
            elif label == "Not Coherent & Not Accurate to the Instructions":
                st.error(f"❌⚠️ **Not Coherent & Not Accurate to the Instructions**\n\n{detail_line}")
            else:
                st.info(f"ℹ️ **{label}**\n\n{detail_line}")

        ticket = st.session_state.get("generated_ticket")
        if ticket:
            st.divider()
            st.subheader("4. Standardized Developer Ticket")
            m1, m2, m3 = st.columns(3)
            m1.metric("Severity", ticket.get("severity", "Medium"))
            m2.metric("Component", ticket.get("component", "General"))
            m3.metric("PRD Status", comb["status_label"] if comb else "Pending")

            st.markdown(f"**Title:** {ticket.get('ticket_title', 'Untitled Ticket')}")
            st.markdown(f"**Probable Root Cause:** {ticket.get('probable_root_cause', 'N/A')}")
            st.markdown(f"**PRD Alignment Notes:** {ticket.get('prd_alignment_notes', 'N/A')}")

            with st.expander("Formatted Ticket Markdown", expanded=True):
                st.markdown(ticket.get("formatted_markdown", ""))


with tab_dashboard:
    st.subheader("QA Manager Dashboard Analytics")
    records = load_triage_records()

    if not records:
        st.info("No triage records logged yet. Run a check in the Triage Workspace to populate metrics.")
    else:
        df = pd.DataFrame(records)
        total_runs = len(df)
        passed_both_count = int((df.get("combined_status") == "Passed Both").sum()) if "combined_status" in df else 0
        avg_p1_ms = round(float(df["phase1_ms"].mean()), 2) if "phase1_ms" in df else 0.0
        avg_p2_ms = round(float(df["phase2_ms"].mean()), 2) if "phase2_ms" in df else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Triaged Reports", total_runs)
        c2.metric("Passed Both Rate", f"{(passed_both_count / total_runs) * 100:.1f}%")
        c3.metric("Avg Phase 1 Speed", f"{avg_p1_ms} ms")
        c4.metric("Avg Phase 2 Speed", f"{avg_p2_ms} ms")

        st.dataframe(df, use_container_width=True)
