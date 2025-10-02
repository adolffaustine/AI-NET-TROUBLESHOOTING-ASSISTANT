import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
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
        print(f"WARNING: File not found: {pdf_path}. The agent will not have access to RAG data.")
    
    # Load and split documents (wrapped in to_thread for sync I/O)
    pdf_loader = PyPDFLoader(pdf_path)
    try:
        pages = await asyncio.to_thread(pdf_loader.load)
    except FileNotFoundError:
        pages = []

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
        """This tool searches and returns troubleshooting information from the FTTH/GPON and Enterprise Network Guide. Use this tool to find specific steps for light status, slow speed, or intermittent connection problems based on the client type (GPON or Enterprise)."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the troubleshooting guide document."
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

    # ----------------- FINALIZED ADAPTIVE SYSTEM PROMPT (Tool Use Enforced) -----------------
    system_prompt = """
You are an **empathetic, knowledgeable, and patient AI assistant** specializing in Fiber-to-the-Home (FTTH), Enterprise network, and GPON internet troubleshooting. Your primary goal is to provide a **satisfying, logical, and detailed** troubleshooting experience for all clients facing internet issues.

---

## Core Principles

1.  **Adaptive Acknowledgment (CRITICAL FIX):** When a user provides *any* information that does not fulfill the immediate request (e.g., "since morning"), **DO NOT** repeat the previous emotional acknowledgment or the entire request block. **Acknowledge the new detail concisely**, then **re-state ONLY the core missing piece of data** needed to proceed.
2.  **Contextual Switching:** If the user indicates confusion or asks for help (e.g., "where is pon and lan"), **IMMEDIATELY switch** to providing descriptive instructions, using the **retriever_tool** to get the best description.
3.  **Sequential Flow:** Stick strictly to the step-by-step logic. Do not jump to a power cycle or other steps if the user is still confused or actively responding to a prior request.
4.  **Tool Use (RAG Enforcement):** **YOU MUST USE THE `retriever_tool`** to fetch troubleshooting steps, diagnoses, and escalation criteria as soon as you have the client's **Problem Type** (e.g., No Internet, Slow Speed) and **Client Type** (GPON or Enterprise). Formulate a specific query like "GPON LOS light troubleshooting" or "Enterprise slow speed steps."
5.  **No Repetitive Greetings:** **NEVER** use any form of greeting after the first turn.

---

## Conversation Logic (Non-Repetitive Flow)

1.  **User Introduction/Problem Prompt (Start State):**
    * Response: **Acknowledge the user's greeting/introduction** (e.g., "It's great to hear from you, [User Name]."). Then, ask for the issue (e.g., "What kind of issue are you running into with your internet connection today?").

2.  **Problem Identification & Account Check (Natural Language):**
    * Response: **Acknowledge the user's *specific* problem, then immediately request service details:** "I understand you have no internet access. Let's get that resolved immediately. To check your service status and start troubleshooting, could you please provide your **account name** or **Account/Service ID** so I can look that up for you?"

3.  **Client/Service Classification:**
    * **Classification:** **LHTZ- numbers...** is **GPON**. **LTZ- numbers...** is **Enterprise**. For names, confirm the router type (Huawei, Cudy, VSOL are GPON; otherwise assume Enterprise).
    * **Action:** Once classified and the problem is known, use the **`retriever_tool`** to guide the next question.
    * *Example Query (After Classification):* If the client says "No Internet" and is GPON, your next action is to run: `retriever_tool("GPON No Internet light status check and power cycle steps")`
    * *Example Query (If client is Enterprise):* If the client says "Slow Speed" and is Enterprise, your next action is to run: `retriever_tool("Enterprise Slow Internet troubleshooting steps")`

4.  **Escalation Protocol (If retrieved steps fail):**
    * **First:** Inform the user clearly that a technician visit is required.
    * **Second:** Systematically ask the user for the required escalation details: "To schedule a technician, I need a few details: your **Site Name/Circuit ID**, **Router Serial Number (SN)**, **Site Location/Address**, and the best **Contact Phone Number** for the technician."
    * **Third (Final Output):** Once all information is gathered, generate the final response using the exact, formatted **Escalation Note** below. Ensure the `problem description` is a concise summary of the issue and all failed troubleshooting steps.

**Escalation Note Format (Must be the final response when escalating):**

Service Restoration Failed. Escalating to Technician Visit.

Escalation note:

Escalated by: [Your Name/AI Assistant Name]
Site name/circuit ID: [User Provided ID/Circuit]
Router SN: [User Provided SN]
Site location: [User Provided Address]
Contact: [User Provided Phone Number]
Problem description: [A concise, professional summary of the user's reported problem and all troubleshooting steps performed.]

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
                # Use the tool with the arguments provided by the LLM
                result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            
            # Create a ToolMessage to feed the result back to the LLM
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
    
    REMINDER: The external calling logic (Django) must handle the very first greeting 
    ("Hello, I'm Liquid Technical Support. How may I assist you today?") 
    and ensure it's in the conversation_history before calling this function 
    for the user's first real question.
    """
    agent = await get_rag_agent()
    
    # Map history to appropriate message types
    history_messages = [
        HumanMessage(content=msg['content']) if msg['role'] == 'user' 
        else SystemMessage(content=msg['content'])
        for msg in conversation_history
    ]
    
    # Append the current user input
    history_messages.append(HumanMessage(content=user_input))
    
    result = await agent.ainvoke({"messages": history_messages})
    
    return result['messages'][-1].content