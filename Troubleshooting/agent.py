import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # <-- FIX IS HERE
from langchain_chroma import Chroma
from langchain_core.tools import tool
import asyncio
from langgraph.graph import StateGraph, END

# Load environment variables once at the top
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please make sure it is set in your .env file.")

# Define a variable to hold the agent instance.
rag_agent = None

async def get_rag_agent():
    """
    Initializes and returns the RAG agent using Google AI services for both
    LLM (Gemini 2.5 Flash) and Embeddings (gemini-embedding-001).
    """
    global rag_agent
    if rag_agent:
        return rag_agent

    # 1. Initialize the LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    
    # 2. Initialize the Google Embeddings Model
    embeddings = GoogleGenerativeAIEmbeddings(
         model="gemini-embedding-001", 
         google_api_key=api_key
    )

    pdf_path = "GPON_Internet_Troubleshooting_Guide.pdf"
    persist_directory = "./gpon_vectorstore" 
    collection_name = "gpon_troubleshooting_guide"

    if not os.path.exists(pdf_path):
        # NOTE: Make sure this PDF file is in the same directory or adjust the path!
        # If running in a web framework, the file path might need to be adjusted relative to the project root.
        print(f"WARNING: File not found: {pdf_path}. The agent will not have access to RAG data.")
    
    # Load and split documents (wrapped in to_thread for sync I/O)
    pdf_loader = PyPDFLoader(pdf_path)
    try:
        pages = await asyncio.to_thread(pdf_loader.load)
    except FileNotFoundError:
        pages = [] # Handle case where PDF isn't found gracefully for testing

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    def create_vectorstore():
        return Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )

    vectorstore = await asyncio.to_thread(create_vectorstore)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 3. Define the Retrieval Tool
    @tool
    def retriever_tool(query: str) -> str:
        """This tool searches and returns the information from the GPON Troubleshooting Guide document."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the GPON Troubleshooting Guide document."
        results = [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)

    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {our_tool.name: our_tool for our_tool in tools}

    # 4. Define LangGraph State and Logic
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

    # ----------------- FINALIZED ADAPTIVE SYSTEM PROMPT -----------------
    system_prompt = """
    You are an **empathetic, knowledgeable, and patient AI assistant** specializing in Fiber-to-the-Home (FTTH) and GPON internet troubleshooting. Your primary goal is to provide a **satisfying, logical, and detailed** troubleshooting experience for residential clients.

    **Core Principles:**
    1.  **Adaptive Acknowledgment:** When a user provides a partial answer (e.g., "POWER is on"), use a concise acknowledgment (e.g., "Understood.") **once** and then immediately ask only for the missing information. **DO NOT** repeat the already confirmed information in the next turn.
    2.  **Contextual Switching:** If the user indicates confusion or asks for help (e.g., "where is pon and lan"), **IMMEDIATELY switch** from asking for the status to providing descriptive instructions to help them locate the equipment or lights. **DO NOT** repeat the technical light names.
    3.  **Sequential Flow:** Stick to the step-by-step logic. Do not jump to a power cycle or other steps if the user is still confused or actively responding to a prior request.
    4.  **No Repetitive Greetings:** **NEVER** use any form of greeting (e.g., "Hello," "Adolf, thank you...") after the first turn.

    **Conversation Logic:**

    1.  **Initial Greeting (First Turn ONLY):**
        * Input: "Initial greeting from Liquid Technical Support."
        * Response: "Hello, I'm Liquid Technical Support, your GPON Troubleshooting Assistant. How may I assist you with your internet connection or equipment today?"

    2.  **User Introduction/Problem Prompt:**
        * **Input:** User provides an introduction (e.g., "hi am adolf").
        * **Response:** "It's great to hear from you, Adolf. What kind of issue are you running into with your internet connection or equipment?"

    3.  **Problem Identification (When Problem is Stated):**
        * **Input:** User states the problem (e.g., "i dont haave internet").
        * **Response:** "I understand you don't have internet access, Adolf. Let's get that resolved. To start, please look at your fiber modem/ONT and tell me the status (color/solid/blinking/off) of the **POWER, PON, LOS, and LAN** lights."

    4.  **Step Progression (Adaptive/Non-Repetitive):**
        * **IF INCOMPLETE DATA:** Acknowledge the known status once, and then ask ONLY for the missing statuses.
            * *Example (User says "power is on"):* "Understood, the POWER light is on. Now, could you please tell me the status (color/solid/blinking/off) of the **PON, LOS, and LAN** lights?"
        * **IF CONFUSION/LOCATION HELP:** Stop asking for the light status. Provide descriptive, helpful guidance.
            * *Example (User says "where is pon and lan"):* "I can help you find those. **PON** and **LOS** are the two lights that show your fiber signal status. They are usually found next to the POWER light on the front of the modem. **LAN** is usually next to the port where the cable goes to your Wi-Fi router. What are the colors of the PON and LOS lights?" (Or guide them to look for the model number to look up a diagram).
        * **IF FULL DATA PROVIDED:** Proceed to the next logical step (e.g., power cycle, cable check, or escalation), using the `retriever_tool` for technical steps.

    5.  **Escalation Protocol (If all steps fail):**
        * If the issue cannot be resolved through client-side troubleshooting, you **must** escalate.
        * **First:** Inform the user clearly that a technician visit is required.
        * **Second:** Systematically ask the user for the required escalation details: "To schedule a technician, I need a few details: your **Site Name/Circuit ID**, **Router Serial Number (SN)**, **Site Location/Address**, and the best **Contact Phone Number** for the technician."
        * **Third (Final Output):** Once all information is gathered, generate the final response using the exact, formatted **Escalation Note** below. Fill the details based on the conversation, and ensure the `problem description` is a concise summary of the issue and all failed troubleshooting steps.

    **Escalation Note Format (Must be the final response when escalating):**

    ```
    Service Restoration Failed. Escalating to Technician Visit.

    Escalation note:

    Escalated by: [Your Name/AI Assistant Name]
    Site name/circuit ID: [User Provided ID/Circuit]
    Router SN: [User Provided SN]
    Site location: [User Provided Address]
    Contact: [User Provided Phone Number]
    Problem description: [A concise, professional summary of the user's reported problem and all troubleshooting steps performed, e.g., "Customer reports 'No Internet'. ONT LOS light is solid red. Power cycle performed (60 seconds) with no change. Issue requires physical site inspection of the fiber termination."]
    ```
    """

    def call_llm(state: AgentState) -> AgentState:
        # Prepend the SystemMessage with the system_prompt for every call
        messages = list(state["messages"])
        messages = [SystemMessage(content=system_prompt)] + messages
        message = llm_with_tools.invoke(messages)
        return {"messages": [message]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] not in tools_dict:
                result = "Incorrect Tool Name, Please Retry and Select tool from list of Available tools."
            else:
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    def return_answer(state: AgentState) -> AgentState:
        return {"messages": [state['messages'][-1]]}

    # 5. Build and Compile LangGraph
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_node("final_answer", return_answer)

    graph.add_conditional_edges(
        "llm", 
        should_continue, 
        {
            True: "retriever_agent",
            False: "final_answer"
        }
    )

    graph.add_edge("retriever_agent", "llm")
    graph.add_edge("final_answer", END)

    graph.set_entry_point("llm")

    rag_agent = graph.compile()
    
    return rag_agent

async def run_gpon_agent_async(user_input, conversation_history):
    """
    Runs the RAG agent with user input and returns the agent's response.
    """
    agent = await get_rag_agent()
    # Map history to appropriate message types
    history_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else SystemMessage(content=msg['content']) 
        for msg in conversation_history
    ]
    
    result = await agent.ainvoke({"messages": history_messages + [HumanMessage(content=user_input)]})
    
    return result['messages'][-1].content