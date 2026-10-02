"""
src/rag_engine.py

RAG Engine for Singapore Silver Support Scheme Streamlit App.
Handles vector database retrieval, live user-uploaded document parsing,
and LLM response generation using Google Gemini and LangChain.
"""

import os
import io
import base64
import logging
from typing import List, Dict, Any, Generator, Optional
import streamlit as st
import pypdf


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Configure Logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_PATH = "./vectorstore/silver_support_chroma"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"  # Uses active Gemini embedding endpoint


def get_api_key() -> str:
    """
    Retrieves the Google Gemini API key securely from Streamlit secrets or environment.
    """
    api_key = None
    try:
        if "API_KEY" in st.secrets:
            api_key = st.secrets["API_KEY"]
        elif "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        logger.debug(f"Streamlit secrets not accessible: {e}")

    if not api_key:
        api_key = os.getenv("API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "API key missing. Please define 'API_KEY' in '.streamlit/secrets.toml' "
            "or set the 'API_KEY' environment variable."
        )

    return api_key


@st.cache_resource(show_spinner=False)
def load_vector_store() -> Optional[Chroma]:
    """
    Loads and caches the persistent Chroma vector store.
    """
    api_key = get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=api_key
    )

    if not os.path.exists(VECTOR_STORE_PATH):
        logger.warning(f"Vector store path '{VECTOR_STORE_PATH}' does not exist.")
        return None

    try:
        vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load Chroma vector store: {e}")
        return None


def extract_text_from_upload(uploaded_file: Any, max_chars: int = 25000) -> Dict[str, Any]:
    """
    Extracts text from PDF/TXT uploads. Captures raw file bytes and MIME type 
    for scanned image PDFs to process directly via Gemini Flash Multimodal.
    """
    if uploaded_file is None:
        return {"content": "", "raw_bytes": None, "mime_type": "", "is_scanned": False}

    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()
    mime_type = uploaded_file.type or "application/pdf"
    extracted_text = ""

    if file_name.endswith(".txt") or mime_type == "text/plain":
        extracted_text = file_bytes.decode("utf-8", errors="ignore")
    elif file_name.endswith(".pdf") or mime_type == "application/pdf":
        try:
            import pypdf
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i, page in enumerate(pdf_reader.pages[:20]):
                t = page.extract_text()
                if t and t.strip():
                    pages_text.append(f"--- Page {i + 1} ---\n{t.strip()}")
            extracted_text = "\n\n".join(pages_text)
        except Exception as e:
            logger.warning(f"pypdf extraction skipped for {uploaded_file.name}: {e}")

    is_scanned = len(extracted_text.strip()) == 0

    return {
        "content": extracted_text.strip() if not is_scanned else "[Scanned Image PDF - Processing via Gemini Vision]",
        "raw_bytes": file_bytes,
        "mime_type": mime_type,
        "is_scanned": is_scanned
    }

def get_gemini_llm(model_id: str, temperature: float = 0.2, top_p: float = 0.95) -> ChatGoogleGenerativeAI:
    """
    Maps user-facing model selection strings to Gemini API model identifiers
    and returns an initialized ChatGoogleGenerativeAI instance.

    Args:
        model_id (str): Selected model key or string identifier.
        temperature (float): Model sampling temperature.
        top_p (float): Model top-p nucleus sampling parameter.

    Returns:
        ChatGoogleGenerativeAI: Configured LangChain model instance.
    """
    api_key = get_api_key()

    # Map user-facing model selection strings to Gemini API identifiers
    model_mapping = {
        "flash 3.8": "gemini-3.8-flash",
        "flash 3.7": "gemini-3.7-flash",
        "flash lite 3.5": "gemini-3.5-flash-lite",
        "flash lite 3.1": "gemini-3.1-flash-lite",
    }

    cleaned_id = model_id.strip().lower()
    mapped_model_id = model_mapping.get(cleaned_id, model_id)

    return ChatGoogleGenerativeAI(
        model=mapped_model_id,
        google_api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        streaming=True
    )


def stream_chat_response(
    messages_history: List[Dict[str, str]],
    uploaded_files_data: List[Dict[str, Any]],
    system_prompt: str,
    model_id: str,
    temperature: float = 0.2,
    top_p: float = 0.95
) -> Generator[str, None, None]:
    """
    Streams response using Gemini Flash. Sends plain text or base64 multimodal 
    payloads for scanned PDFs and images.
    """
    latest_query = messages_history[-1]["content"] if messages_history else ""

    # 1. Retrieve Vector Context
    vector_context = ""
    vector_store = load_vector_store()
    if vector_store and latest_query:
        try:
            results = vector_store.similarity_search(latest_query, k=3)
            retrieved_chunks = [f"[Policy Context {i+1}]:\n{doc.page_content}" for i, doc in enumerate(results)]
            vector_context = "\n\n".join(retrieved_chunks)
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")

    # 2. System Instruction Setup
    system_instruction = (
        f"{system_prompt}\n\n"
        f"=== OFFICIAL POLICY DATABASE CONTEXT ===\n"
        f"{vector_context if vector_context else 'No official database context found.'}\n\n"
        f"Instructions: Review all attached documents (including scanned image PDFs) "
        f"to extract key details like Notice of Assessment (NOA) income, dates, or personal records."
    )

    langchain_messages = [SystemMessage(content=system_instruction)]

    # 3. Add History
    for msg in messages_history[:-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))

    # 4. Construct Multimodal Payload for Active User Turn
    human_payload = []

    if uploaded_files_data:
        for doc in uploaded_files_data:
            fname = doc.get("name", "Attached File")
            text = doc.get("content", "")
            raw_bytes = doc.get("raw_bytes")
            mime_type = doc.get("mime_type", "application/pdf")
            is_scanned = doc.get("is_scanned", False)

            if not is_scanned and text:
                human_payload.append({
                    "type": "text",
                    "text": f"=== ATTACHED TEXT FILE: {fname} ===\n{text}\n"
                })
            elif raw_bytes:
                # Multimodal payload sent to Gemini Flash for scanned images/PDFs
                b64_data = base64.b64encode(raw_bytes).decode("utf-8")
                human_payload.append({
                    "type": "text",
                    "text": f"=== ATTACHED SCANNED FILE ({fname}) ==="
                })
                human_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                })

    human_payload.append({
        "type": "text",
        "text": f"=== USER QUESTION ===\n{latest_query}"
    })

    langchain_messages.append(HumanMessage(content=human_payload))

    # 5. Stream Response
    try:
        llm = get_gemini_llm(model_id=model_id, temperature=temperature, top_p=top_p)
        for chunk in llm.stream(langchain_messages):
            if chunk.content:
                if isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk.content, list):
                    for block in chunk.content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
                        elif isinstance(block, str):
                            yield block
    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        yield f"\n\n[Error generating response: {str(e)}]"

def generate_diagnostic_explanation(
    eval_result: Dict[str, Any],
    user_inputs: Dict[str, Any],
    model_id: str,
    temperature: float = 0.2,
    top_p: float = 0.95
) -> str:
    """
    Generates a personalized diagnostic explanation for eligibility outcomes or payout tiers.
    """
    failed_reasons = eval_result.get("failed_criteria", [])
    passed_reasons = eval_result.get("passed_criteria", [])
    is_eligible = eval_result.get("is_eligible", False)
    payout = eval_result.get("payout_amount", 0.0)
    pchhi = eval_result.get("pchhi", 0.0)

    search_query = " ".join(failed_reasons) if failed_reasons else f"Silver Support payout for PCHHI ${pchhi} in {user_inputs.get('hdb_flat_type')}"
    
    vector_store = load_vector_store()
    retrieved_policy = ""
    if vector_store:
        try:
            docs = vector_store.similarity_search(search_query, k=3)
            retrieved_policy = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Diagnostic retrieval failed: {e}")

    diagnostic_prompt = f"""
You are an empathetic, expert Singapore Senior Policy Consultant specializing in the Silver Support Scheme (2025/2026 guidelines).

APPLICANT PROFILE:
- Age: {user_inputs.get('age')}
- Total CPF at 55: ${user_inputs.get('total_cpf_contributions_at_55', 0):,.2f}
- Self-Employed / Platform: {user_inputs.get('is_self_employed_or_platform')}
- Average Net Trade Income: ${user_inputs.get('net_trade_income_avg', 0):,.2f}
- Flat Type: {user_inputs.get('hdb_flat_type')}
- Applicant Owns Private/Multiple Properties: {user_inputs.get('owns_private_or_multiple_properties')}
- Spouse Owns Private/Multiple Properties: {user_inputs.get('spouse_owns_private_or_multiple_properties')}
- Monthly Household Income: ${user_inputs.get('monthly_household_income', 0):,.2f}
- Household Members: {user_inputs.get('household_members_count')}
- Calculated PCHHI: ${pchhi:,.2f}
- Receiving ComCare LTA: {user_inputs.get('on_comcare_lta')}

EVALUATION OUTCOME:
- Overall Eligible: {is_eligible}
- Calculated Quarterly Payout: ${payout:,.2f}
- Criteria Met: {passed_reasons}
- Criteria Failed: {failed_reasons}

RELEVANT POLICY GUIDELINES:
{retrieved_policy}

TASK:
Write a clear, supportive, and structured explanation for the senior or their caregiver:
1. **Outcome Summary & Specific Reasons**: Explain clearly why the applicant is ineligible OR why they qualify for their specific payout tier (${payout}/quarter). Reference exact figures (such as income, flat type, PCHHI of ${pchhi:,.2f}, or property ownership status).
2. **Actionable Checklist**: Provide a step-by-step checklist on what the applicant or family can do next (e.g., how to request an eligibility review with the CPF Board, update household records with ICA, or explore alternative assistance like ComCare if facing financial difficulty).

Format with clear headers and bullet points. Keep the tone warm, respectful, and authoritative.
"""

    try:
        llm = get_gemini_llm(model_id=model_id, temperature=temperature, top_p=top_p)
        response = llm.invoke([
            SystemMessage(content="You are a helpful Singapore Government Policy Advisor."),
            HumanMessage(content=diagnostic_prompt)
        ])
        
        # Extract text from content blocks if response.content is a list/dict (Fixes Page 1 issue)
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            extracted = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
            return "".join(extracted) if extracted else str(content)
        return str(content)

    except Exception as e:
        logger.error(f"Failed to generate diagnostic explanation: {e}")
        return (
            f"### Outcome Explanation\n\n"
            f"**Eligibility Status**: {'Eligible' if is_eligible else 'Ineligible'}\n"
            f"**Quarterly Payout**: ${payout:,.2f}\n"
            f"**Calculated PCHHI**: ${pchhi:,.2f}\n\n"
            f"**Key Factors**:\n"
            + "\n".join([f"- {f}" for f in (failed_reasons if failed_reasons else passed_reasons)])
        )