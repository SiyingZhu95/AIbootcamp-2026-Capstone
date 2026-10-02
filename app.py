"""
app.py

Main Streamlit application entry point for the Singapore Silver Support Scheme Assistant.
Handles base page configuration, session state initialization, sidebar controls,
and tab navigation routing.
"""

import streamlit as st
from src.rules import evaluate_silver_support_eligibility
from src.rag_engine import (
    generate_diagnostic_explanation,
    extract_text_from_upload,
    stream_chat_response,
)

# ------------------------------------------------------------------------------
# Password Gate
# ------------------------------------------------------------------------------
def check_password():
    """Returns True if the user enters the correct password."""
    def password_entered():
        if st.session_state.get("password_input") == "imda":
            st.session_state["password_correct"] = True
            if "password_input" in st.session_state:
                del st.session_state["password_input"]  # Remove stored password string
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct"):
        return True

    # Password prompt interface
    st.markdown("## 🔒 Access Restricted")
    st.caption("Please authenticate to view the Silver Support Scheme Assistant.")
    
    st.text_input(
        "Enter Access Password:",
        type="password",
        on_change=password_entered,
        key="password_input"
    )

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("❌ Incorrect password. Please try again.")

    return False

# Stop execution if password is not verified
if not check_password():
    st.stop()

# 1. Page Configuration
st.set_page_config(
    page_title="Silver Support Scheme Assistant",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disclaimer Expander
with st.expander("⚠️ Disclaimer & Important Notice", expanded=False):
    st.warning(
        "IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. "
        "The information provided here is NOT intended for real-world usage and should not be relied upon "
        "for making any decisions, especially those related to financial, legal, or healthcare matters.\n\n"
        "Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. "
        "You assume full responsibility for how you use any generated output.\n\n"
        "Always consult with qualified professionals for accurate and personalised advice."
    )

# 2. Session State Initialization
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    # Default model set to flash lite 3.5
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "flash lite 3.5"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2

    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.95

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are an empathetic, accurate Singapore Social Policy Advisor. "
            "Explain official policies clearly and concisely using retrieved policy "
            "context and any user-uploaded reference documents."
        )


init_session_state()


# 3. Sidebar Navigation & Controls
st.sidebar.title("🇸🇬 Silver Support Assistant")
st.sidebar.markdown(
    """
    **Empowering Seniors, Caregivers & Social Workers**
    
    Automated eligibility evaluations, diagnostic appeal guidance, and AI-powered 
    policy Q&A for the **Singapore Silver Support Scheme**.
    """
)
st.sidebar.divider()

st.sidebar.divider()

# Advanced Settings Expander
with st.sidebar.expander("⚙️ Advanced Settings"):
    model_options = ["flash 3.8", "flash 3.7", "flash lite 3.5", "flash lite 3.1"]
    
    current_model_index = (
        model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_options
        else 2
    )

    st.session_state.selected_model = st.selectbox(
        "Gemini Model Variant",
        options=model_options,
        index=current_model_index,
        help="Select the Gemini model variant to power RAG chat and diagnostic engines."
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.temperature),
        step=0.05,
    )

    st.session_state.top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.top_p),
        step=0.05,
    )

    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=140,
    )

# Clear Chat History Button (placed in sidebar)
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.uploaded_docs = []
    st.rerun()


# 4. Tab 1 Implementation
def render_tab1():
    """Renders Use Case 1: 'Why Didn't I Qualify?' Diagnostic & Appeal Assistant."""
    st.header("📋 Diagnostic & Eligibility Appeal Assistant")
    st.caption("Evaluate senior profile criteria against 2025/2026 guidelines and generate personalized diagnostic explanations.")

    with st.container(border=True):
        st.subheader("👴 Senior Profile & Household Details")
        col1, col2, col3 = st.columns(3)

        # Column 1: Personal & Employment Details
        with col1:
            st.markdown("**Personal Details**")
            age = st.number_input("Senior's Age", min_value=0, max_value=120, value=65, step=1)
            total_cpf = st.number_input("Total CPF Contributions at Age 55 ($)", min_value=0.0, value=0.0, step=5000.0, format="%.2f")
            
            is_self_employed = st.checkbox("Self-Employed / Platform Worker")
            net_trade_income = 0.0
            if is_self_employed:
                net_trade_income = st.number_input(
                    "Avg Annual Net Trade Income ($)",
                    min_value=0.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f",
                    help="Average net trade income earned between ages 45 and 54."
                )

        # Column 2: Property & Housing
        with col2:
            st.markdown("**Housing & Property Ownership**")
            flat_type = st.selectbox(
                "HDB Flat Type",
                options=["1-2 Room", "3-Room", "4-Room", "5-Room Live-In", "5-Room Owned / Private"],
                index=1
            )
            owns_pvt = st.checkbox("Applicant owns private, 5-room, or multiple properties")
            spouse_owns_pvt = st.checkbox("Spouse owns private, 5-room, or multiple properties")

        # Column 3: Household Financials & Assistance
        with col3:
            st.markdown("**Household Financials**")
            monthly_income = st.number_input("Gross Monthly Household Income ($)", min_value=0.0, value=0.0, step=500.0, format="%.2f")
            members_count = st.number_input("Number of Household Members", min_value=1, max_value=20, value=1, step=1)
            on_comcare_lta = st.checkbox("Receiving ComCare Long-Term Assistance (LTA)")

        st.divider()
        run_assessment = st.button("🔍 Run Diagnostic Assessment", type="primary", use_container_width=True)

    # Assessment Processing on Button Click
    if run_assessment:
        user_data = {
            "age": age,
            "total_cpf_contributions_at_55": total_cpf,
            "is_self_employed_or_platform": is_self_employed,
            "net_trade_income_avg": net_trade_income if is_self_employed else 0.0,
            "hdb_flat_type": flat_type,
            "owns_private_or_multiple_properties": owns_pvt,
            "spouse_owns_private_or_multiple_properties": spouse_owns_pvt,
            "monthly_household_income": monthly_income,
            "household_members_count": members_count,
            "on_comcare_lta": on_comcare_lta
        }

        # 1. Rule Evaluation
        eval_result = evaluate_silver_support_eligibility(user_data)

        # 2. Display Metrics & Status
        st.divider()
        st.subheader("📊 Assessment Summary")

        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            if eval_result["is_eligible"]:
                st.success("### STATUS: ELIGIBLE 🎉")
            else:
                st.error("### STATUS: INELIGIBLE ❌")

        with m_col2:
            st.metric(
                label="Estimated Quarterly Payout",
                value=f"${eval_result['payout_amount']:,.2f} / qtr"
            )

        with m_col3:
            st.metric(
                label="Per Capita Household Income (PCHHI)",
                value=f"${eval_result['pchhi']:,.2f} / mth"
            )

        # Detailed Criteria Breakdown
        with st.expander("🔍 View Assessment Criteria Breakdown", expanded=not eval_result["is_eligible"]):
            if eval_result["passed_criteria"]:
                st.markdown("**Criteria Met:**")
                for c in eval_result["passed_criteria"]:
                    st.markdown(f"- ✅ {c}")

            if eval_result["failed_criteria"]:
                st.markdown("**Criteria Failed:**")
                for c in eval_result["failed_criteria"]:
                    st.markdown(f"- ❌ {c}")

        # 3. RAG Diagnostic Explanation & Review Guidance
        st.divider()
        st.subheader("💡 Diagnostic Analysis & Appeal Action Plan")

        with st.spinner("Analyzing policy guidelines and generating personalized diagnostic review..."):
            explanation = generate_diagnostic_explanation(
                eval_result=eval_result,
                user_inputs=user_data,
                model_id=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p
            )

        with st.container(border=True):
            st.markdown(explanation)

def render_tab2():
    """Renders Use Case 4: Social Worker Policy Explainer & Document Assistant."""
    st.header("💬 Policy Explainer & Document Assistant")
    st.caption("Ask complex policy questions or upload document references for plain-language explanations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []

    # 1. Active Loaded Document Badges (At top of assistant page)
    if st.session_state.uploaded_docs:
        st.markdown("**📄 Active Reference Documents:**")
        docs_to_remove = []
        for idx, doc in enumerate(st.session_state.uploaded_docs):
            d_col1, d_col2 = st.columns([4, 1])
            status_str = "Scanned Image / PDF" if doc.get("is_scanned") else f"{len(doc['content']):,} chars extracted"
            with d_col1:
                st.success(f"Loaded: **{doc['name']}** ({status_str})")
            with d_col2:
                if st.button("❌ Remove", key=f"remove_{doc['id']}", use_container_width=True):
                    docs_to_remove.append(idx)

        if docs_to_remove:
            for idx in reversed(docs_to_remove):
                st.session_state.uploaded_docs.pop(idx)
            st.rerun()

    st.divider()

    # 2. Main Chat Conversation Area
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 3. Quick Suggestion Chips
    st.markdown("**Quick Inquiry Suggestions:**")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    suggested_prompt = None

    with chip_col1:
        if st.button("How is household income calculated if my adult child moves out?", key="chip1", use_container_width=True):
            suggested_prompt = "How is household income calculated if my adult child moves out?"

    with chip_col2:
        if st.button("How are self-employed lifetime earnings evaluated for Silver Support?", key="chip2", use_container_width=True):
            suggested_prompt = "How are self-employed lifetime earnings evaluated for Silver Support?"

    with chip_col3:
        if st.button("When are quarterly payouts credited and how does PayNow-NRIC work?", key="chip3", use_container_width=True):
            suggested_prompt = "When are quarterly payouts credited and how does PayNow-NRIC work?"

    # 4. Document Uploader (Placed below conversation history and chips, above chat input)
    st.write("")
    uploaded_files = st.file_uploader(
        "📎 Attach Reference Document (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="user_doc_uploader",
        help="Upload Notice of Assessment (NOA), case notes, or scanned PDFs."
    )

    if uploaded_files:
        existing_ids = {doc["id"] for doc in st.session_state.uploaded_docs}
        new_doc_added = False
        for file in uploaded_files:
            file_id = f"{file.name}_{file.size}"
            if file_id not in existing_ids:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        extracted_info = extract_text_from_upload(file)
                        st.session_state.uploaded_docs.append({
                            "id": file_id,
                            "name": file.name,
                            "filename": file.name,
                            "content": extracted_info["content"],
                            "raw_bytes": extracted_info["raw_bytes"],
                            "mime_type": extracted_info["mime_type"],
                            "is_scanned": extracted_info["is_scanned"]
                        })
                        new_doc_added = True
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
        if new_doc_added:
            st.rerun()

    # 5. Chat Input Box & Generation Handling
    chat_input_text = st.chat_input("Ask a policy question or query uploaded documents...")
    prompt = suggested_prompt or chat_input_text

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_generator = stream_chat_response(
                    messages_history=st.session_state.messages,
                    uploaded_files_data=st.session_state.uploaded_docs,
                    system_prompt=st.session_state.system_prompt,
                    model_id=st.session_state.selected_model,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p
                )
                full_response = st.write_stream(response_generator)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

def render_about_us():
    """Renders the Project Overview and About Us page."""
    st.header("ℹ️ About Us & Project Overview")
    st.caption("AI-Powered Decision Support & Policy Explainer for Social Workers and Case Officers")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🎯 Project Scope & Objectives")
        st.markdown(
            """
            Public assistance frameworks like the **Silver Support Scheme** involve multi-layered eligibility rules, 
            housing criteria, and income threshold evaluations. Social workers often need to interpret dense policy manuals 
            while manually extracting information from client documents (such as IRAS Notice of Assessment slips or CPF statements).

            **Key Objectives:**
            * **Reduce Administrative Burden**: Automate manual document reading and plain-language summarization.
            * **Policy Accessibility**: Provide instant, grounded semantic search across official policy guidelines.
            * **Accurate Client Assessment**: Evaluate uploaded client documents (including scanned image PDFs) directly against eligibility thresholds.
            * **Trust & Transparency**: Eliminate hallucination by forcing strict context-grounded retrieval (RAG) with source attribution.
            """
        )

    with col2:
        st.subheader("💡 Core Features")
        st.markdown(
            """
            * **Semantic Vector RAG Search**: Instant policy retrieval using ChromaDB.
            * **Multimodal Vision Ingestion**: OCR capabilities powered by Gemini Flash for scanned PDFs and image files.
            * **Interactive Explainer**: Conversational assistant capable of handling multi-turn follow-up queries.
            * **Real-time Streaming**: Instant tokens generation for high-responsiveness case work.
            """
        )

    st.markdown("---")

    st.subheader("📚 Data Sources & Integration Framework")

    ds_col1, ds_col2, ds_col3 = st.columns(3)

    with ds_col1:
        st.markdown("#### 🏛️ Public Policy Manuals")
        st.markdown(
            """
            * **Silver Support Scheme Guidelines**
            """
        )

    with ds_col2:
        st.markdown("#### 📄 Client Reference Documents")
        st.markdown(
            """
            * **IRAS Notice of Assessment (NOA)**
            * **Payslips & CPF Statements**
            * **Medical & Case Social Notes**
            * **Or other relevant documents**
            """
        )

    with ds_col3:
        st.markdown("#### ⚡ Technology Stack")
        st.markdown(
            """
            * **LLM Engine**: Google Gemini Flash (Vision Enabled)
            * **Vector Database**: ChromaDB
            * **Embeddings**: Google Generative AI Embeddings
            * **Interface**: Streamlit & Graphviz
            """
        )

def render_methodology():
    """Renders the technical architecture, data pipeline, and workflow diagrams."""
    st.header("📐 Technical Methodology & Process Flows")
    st.caption("Detailed overview of technical data pipelines and architectural workflows.")

    st.markdown("---")

    st.subheader("⚙️ System Architecture Overview")
    st.markdown(
        """
        The application is structured into two core operational pathways:
        1. **Policy RAG Pipeline**: Embeds official guidelines into ChromaDB and retrieves relevant context chunks dynamically.
        2. **Multimodal Document Pipeline**: Parses structured text via `pypdf` and falls back seamlessly to Gemini's visual OCR engine for scanned files.
        """
    )

    st.markdown("---")

    # Flowchart for Use Case 1: Policy Retrieval & Querying
    st.subheader("🔄 Flowchart 1: Policy Search & Knowledge Retrieval Pipeline")
    st.caption("How policy questions are embedded, retrieved from ChromaDB, and processed by the LLM.")

    flowchart_uc1 = """
    digraph policy_rag {
        rankdir=LR;
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];
        
        user [label="User Query", fillcolor="#D1E8FF"];
        embed [label="Generate Query Embedding\n(text-embedding-004)", fillcolor="#E2F0D9"];
        vector_db [label="ChromaDB Vector Store\n(Similarity Search k=3)", fillcolor="#FFF2CC"];
        chunks [label="Retrieved Policy Chunks", fillcolor="#FFF2CC"];
        sys_prompt [label="System Prompt Assembly\n(Strict Grounding Instruction)", fillcolor="#E2F0D9"];
        llm [label="Gemini Flash Engine", fillcolor="#FCE4D6"];
        response [label="Streamed Policy Response", fillcolor="#D1E8FF"];

        user -> embed;
        embed -> vector_db;
        vector_db -> chunks;
        chunks -> sys_prompt;
        sys_prompt -> llm;
        llm -> response;
    }
    """
    st.graphviz_chart(flowchart_uc1, use_container_width=True)

    st.markdown("---")

    # Flowchart for Use Case 2: Multimodal Document Parsing & Assistant
    st.subheader("🔄 Flowchart 2: Document Processing & Interactive Assistant Pipeline")
    st.caption("How uploaded PDF/TXT files are parsed, checked for scanned images, and processed with conversation history.")

    flowchart_uc2 = """
    digraph doc_assistant {
        rankdir=LR;
        graph [pad="0.2", nodesep="0.4", ranksep="0.5"];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];
        
        upload [label="User Uploads Document\n(PDF / TXT)", fillcolor="#D1E8FF"];
        parse_attempt [label="Extract Text with pypdf / txt", fillcolor="#E2F0D9"];
        check_scanned [label="Is Text\nExtracted?", shape=diamond, style=filled, fillcolor="#FFF2CC"];
        
        text_payload [label="Format Plain Text Block\n(Injected into Prompt)", fillcolor="#E2F0D9"];
        vision_payload [label="Encode Raw Bytes to Base64\n(Multimodal Image Payload)", fillcolor="#FCE4D6"];
        
        assemble [label="Combine History\n& User Question", fillcolor="#E2F0D9"];
        llm [label="Gemini Flash\nMultimodal LLM", fillcolor="#FCE4D6"];
        output [label="Streamed Assistant Response\n(With Document Insights)", fillcolor="#D1E8FF"];

        upload -> parse_attempt;
        parse_attempt -> check_scanned;
        check_scanned -> text_payload [label="Yes"];
        check_scanned -> vision_payload [label="No (Scanned)"];
        text_payload -> assemble;
        vision_payload -> assemble;
        assemble -> llm;
        llm -> output;
    }
    """
    st.graphviz_chart(flowchart_uc2, use_container_width=True)

    st.markdown("---")

    # Implementation Matrix Table
    st.subheader("📊 Implementation Pipeline Matrix")
    
    matrix_data = {
        "Component": ["Vector Embeddings", "Vector Storage", "PDF Text Extractor", "Scanned PDF Processor", "LLM Orchestration"],
        "Technology / Package": ["Google GenAI Embeddings", "ChromaDB", "pypdf", "Gemini Native Multimodal Vision", "LangChain / Google GenAI SDK"],
        "Fallback Mechanism": ["N/A", "In-Memory Temporary Store", "Gemini Vision Base64 OCR", "Plain-text context insertion", "Exception Error Handler"]
    }
    st.table(matrix_data)

# Main Page Navigation
tab_policy, tab_assistant, tab_about, tab_methodology = st.tabs([
    "🏛️ Diagnostic & Appeal Assistant",
    "💬 Policy Explainer & Assistant",
    "ℹ️ About Us",
    "📐 Methodology"
])

with tab_policy:
    render_tab1()

with tab_assistant:
    render_tab2()

with tab_about:
    render_about_us()

with tab_methodology:
    render_methodology()