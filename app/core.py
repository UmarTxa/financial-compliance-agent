import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END # LangGraph for the Agent
from langfuse.langchain import CallbackHandler

print("--- Initializing Financial Agent Core ---")

# --- 0. Pydantic Schema (Structured Output Tool) ---
class ComplianceSummary(BaseModel):
    """Structured data extracted for financial and compliance analysis."""
    company_name: str = Field(description="The full legal name of the company.")
    fiscal_year: int = Field(description="The reporting year of the document (e.g., 2024).")
    has_coi_policy: bool = Field(description="True if a Conflict of Interest policy is explicitly mentioned, False otherwise.")
    total_revenue_text: str = Field(description="The raw text string of the total reported revenue (e.g., '$4.2 Billion').")

# --- 1. Agent State (Memory) ---
class FinancialAgentState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    structured_data: ComplianceSummary # Will hold the Pydantic model instance
    final_report: str

# --- 2. Configuration and Tool Setup ---
# Check for API Key
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set it in your terminal.")

# Initialize LLM (Used by all nodes)
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
# --- 5. LANGFUSE TRACING INITIALIZATION ---
# This handler will automatically capture all LangChain/LangGraph events
# Note: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST should be set as environment variables
langfuse_handler = CallbackHandler()
# We create a configuration dictionary to pass to the graph
# This tells every node in the graph to use the LangFuse callback
graph_config = {"callbacks": [langfuse_handler]}

print("LangFuse Tracing Handler Initialized.")
extraction_tool = convert_to_openai_tool(ComplianceSummary) # Tool for LLM structured extraction

# --- 3. RAG Setup (Build the Vector Store) ---
pdf_path = pdf_path = r"C:\Users\User\financial_agent\data\unilever-annual-report-and-accounts-2024.pdf" # <-- REMEMBER TO USE YOUR FILE'S NAME!
print(f"Loading and indexing {pdf_path}...")

# 3a. Load & Split
loader = PyPDFLoader(pdf_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)

# 3b. Embed & Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(k=5) # Retrieve 5 top chunks for extraction accuracy

print("Vector Store and Tools Ready.")

from langchain_core.runnables import RunnableConfig

# --- 5. AGENT NODE FUNCTIONS ---

# A. RAG RETRIEVER NODE (Updates 'retrieved_docs')
def rag_retriever(state: FinancialAgentState, config: RunnableConfig) -> dict:
    """Retrieves relevant document chunks based on the user's question."""
    print("--- Executing A: RAG Retriever Node ---")
    question = state["question"]
    
    # Invoke the retriever object created in Action 2
    retrieved_docs = retriever.invoke(question) 
    
    # Return the update dictionary to write results into the state
    return {"retrieved_docs": retrieved_docs}

    # app/core.py (Continued)

# B. CHECK RETRIEVAL (The Guardrail - Determines next step)
def check_retrieval(state: FinancialAgentState) -> str:
    """Routes based on whether documents were found (Guardrail)."""
    
    # Checks if the retrieved_docs list is present and non-empty
    if state.get("retrieved_docs") and len(state["retrieved_docs"]) > 0:
        print("Guardrail Check: Context found. Routing to Data Extraction.")
        return "data_extractor"
    else:
        # If no documents, we skip extraction and go straight to synthesizing a failure report
        print("Guardrail Check: No context found. Routing to Synthesizer for graceful response.")
        return "final_synthesizer"

        # app/core.py (Continued)

# C. DATA EXTRACTOR NODE (Updates 'structured_data')
def data_extractor(state: FinancialAgentState, config: RunnableConfig) -> dict:
    """Uses LLM with Pydantic Tool to extract structured data from retrieved text."""
    print("--- Executing C: Data Extractor Node (Structured Output) ---")
    docs = state["retrieved_docs"]
    
    # Combine documents into a single context string for the LLM
    context = "\n\n".join(doc.page_content for doc in docs)

    # 1. Define Extraction Prompt
    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial analyst. Extract ALL required data fields from the context and use the provided tool call."),
        ("human", f"Context to analyze:\n\n{context}\n\nExtract the requested structured financial and compliance data."),
    ])

    # 2. Bind the Pydantic tool to the LLM (Forces JSON output)
    structured_extractor = extractor_prompt | llm.bind_tools(tools=[extraction_tool])
    
    # 3. Invoke and Parse the Tool Call result
    try:
        result = structured_extractor.invoke({}) 
        
        # Extract the dictionary of arguments used for the tool call
        extracted_data_dict = result.additional_kwargs["tool_calls"][0]["function"]["arguments"]
        
        # 4. Convert the dictionary back to our Pydantic model for type safety
        # This is where type checking and validation occurs!
        structured_data = ComplianceSummary(**extracted_data_dict)
        
    except Exception as e:
        print(f"ERROR in extraction: {e}")
        # If extraction fails, we return a failure message to be passed to the synthesizer
        return {"final_report": f"Extraction failed due to internal LLM error: {e}"} 
    
    return {"structured_data": structured_data}

    # app/core.py (Continued)

# D. COMPLIANCE CHECK NODE (Updates 'final_report' with findings)
def compliance_check(state: FinancialAgentState, config: RunnableConfig) -> dict:
    """Performs deterministic logic on the structured data and generates findings."""
    print("--- Executing D: Compliance Check (Python Logic) ---")
    
    # --- CRITICAL FIX: Add check for missing structured_data ---
    if not state.get("structured_data"):
        print("FATAL: Structured data is NULL. Cannot proceed with audit.")
        # We update final_report with an immediate failure message
        return {"final_report": "AUDIT FAILED: The LLM could not extract structured data. Please check the document content or the model's output for parsing errors."}
        
    data = state["structured_data"] # Now we know 'data' is a Pydantic object

    findings = []
    
    # 1. Auditing the PII/COI Policy (The line that previously crashed)
    if not data.has_coi_policy:
        findings.append("FAILURE: Conflict of Interest (COI) policy was NOT explicitly confirmed in the document. HIGH RISK.")
    else:
        findings.append("SUCCESS: COI policy is confirmed and mentioned.")

    # 2. Auditing the Revenue Format
    if "$" not in data.total_revenue_text:
         findings.append(f"WARNING: Raw Revenue Text ('{data.total_revenue_text}') did not contain expected currency ($). Potential normalization issue for auditing.")
    else:
         findings.append(f"AUDIT OK: Raw Revenue Text is: {data.total_revenue_text}")
        
    # Final Finding String
    return {"final_report": "\n".join(findings)}

# E. FINAL SYNTHESIZER NODE (Compiles the final human-readable answer)
def final_synthesizer(state: FinancialAgentState, config: RunnableConfig) -> dict:
    """Synthesizes the final human-readable report."""
    print("--- Executing E: Final Synthesizer Node ---")
    
    # 1. HANDLE FAILURE FROM ANY PREVIOUS NODE
    # Check if the report was set to a failure message earlier (e.g., from compliance_check)
    if state["final_report"] and ("Extraction failed" in state["final_report"] or "ERROR" in state["final_report"]):
        return {"final_report": state["final_report"]}

    # --- CRITICAL FIX ---
    # Check if structured_data is actually a Pydantic object before dumping it.
    if not state.get("structured_data") or not isinstance(state["structured_data"], BaseModel):
        print("CRITICAL WARNING: No valid Pydantic model found for synthesis. Returning raw findings.")
        return {"final_report": f"Synthesis Error: Structured data extraction failed. Raw findings: {state.get('final_report', 'None')}"}
    # ------------------
        
    # 2. Prepare input for the LLM
    summary_input = {
        "question": state["question"],
        # This line is safe now because we checked the type
        "extracted_data": state["structured_data"].model_dump_json(indent=2), 
        "compliance_findings": state["final_report"] 
    }
    
    synthesizer_prompt = ChatPromptTemplate.from_template(
        """You are a senior financial reporter. Based on the user's question,
        summarize the extracted data and the compliance findings into a formal,
        brief report. State the compliance findings clearly.

        USER QUESTION: {question}
        
        --- EXTRACTED STRUCTURED DATA (for reference) ---
        {extracted_data}
        
        --- COMPLIANCE FINDINGS (from Audit) ---
        {compliance_findings}
        
        --- FINAL REPORT ---
        """
    )
    
    # 3. Generate the final report
    final_chain = synthesizer_prompt | llm | StrOutputParser()
    final_text = final_chain.invoke(summary_input)
    
    return {"final_report": final_text}
    # app/core.py (Continued)

# --- 6. ASSEMBLE THE GRAPH ---
print("Building Financial Compliance Agent Graph...")
workflow = StateGraph(FinancialAgentState)

# 1. Add Nodes
workflow.add_node("rag_retriever", rag_retriever)
workflow.add_node("data_extractor", data_extractor)
workflow.add_node("compliance_check", compliance_check)
workflow.add_node("final_synthesizer", final_synthesizer)

# 2. Define Edges (The Flow)
workflow.set_entry_point("rag_retriever") # Start the process at retrieval

# Conditional Edge: Retrieval Guardrail (Bridges A -> C or B)
# If documents found, go to extraction; otherwise, go straight to synthesizer.
# Conditional Edge: Retrieval Guardrail
workflow.add_conditional_edges(
    "rag_retriever", # Source Node (positional argument)
    check_retrieval, # Routing Function (positional argument)
    {
        "data_extractor": "data_extractor", 
        "final_synthesizer": "final_synthesizer", 
    }
)

# Sequential Edges: The main processing chain
workflow.add_edge("data_extractor", "compliance_check")
workflow.add_edge("compliance_check", "final_synthesizer")

# Final Edge: All successful paths must END
workflow.add_edge("final_synthesizer", END)

# Compile the runnable graph
agent_graph = workflow.compile()
print("--- Financial Compliance Agent Graph Ready! ---")