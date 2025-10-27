from fastapi import FastAPI
from pydantic import BaseModel
from app.core import agent_graph, FinancialAgentState, graph_config # Import the finished graph

# Initialize our FastAPI app
app = FastAPI(
    title="Financial Compliance Agent API",
    description="Multi-step agent for extracting structured financial data and running compliance checks.",
)

# Define the structure of an incoming question
class QueryRequest(BaseModel):
    question: str

# Define the API endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Receives a question, passes it to the multi-step agent graph,
    and returns the final report.
    """
    question = request.question
    print(f"\n--- API Received Question: {question} ---")
    
    # Create the initial state dictionary
    inputs: FinancialAgentState = {"question": question, "retrieved_docs": [], "structured_data": None, "final_report": ""} # Initialize empty state

    # Run the graph! This executes all nodes sequentially
    result_state = agent_graph.invoke(inputs, config=graph_config)
    
    # The final output is stored in the 'final_report' key of the state
    response = result_state.get("final_report", "Agent failed to generate a report.")
    
    print(f"--- API Returning Final Report ---\n")
    
    return {"answer": response}

# Add a simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Financial Compliance Agent! Go to /docs to test the API."}