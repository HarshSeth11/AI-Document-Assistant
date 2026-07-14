import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
from app.retrieval_service import retrieve_with_confidence
from app.models import Document
from app.analysis_service import summarize_document, compare_documents

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)

# ── Tool schema for the LLM ──
rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search uploaded documents for relevant information to answer the user's question. Use this whenever the question could be answered from company documents, policies, or uploaded files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — rephrase the user's question into clear search terms"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_documents",
            "description": "Get a list of all documents currently uploaded and available to search.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": "Generate a summary of an entire document. Use this when the user asks to summarize, give an overview, or explain what a whole document is about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {
                        "type": "string",
                        "description": "The filename or partial name of the document to summarize"
                    }
                },
                "required": ["document_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": "Compare two documents and identify similarities and differences. Use this when the user asks to compare, contrast, or find differences between two documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_a": {
                        "type": "string",
                        "description": "Filename or partial name of the first document"
                    },
                    "document_b": {
                        "type": "string",
                        "description": "Filename or partial name of the second document"
                    }
                },
                "required": ["document_a", "document_b"]
            }
        }
    }
]

llm_with_tools = llm.bind_tools(rag_tools)

# ── State — carries db session through the graph ──
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    db_session: Session
    terminal_tool_used: bool  # NEW — true once summarize/compare succeeds

llm_no_tools = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0
)  # note: NOT bound to tools

def agent_final_node(state: AgentState) -> AgentState:
    messages = [m for m in state["messages"]]
    response = llm_no_tools.invoke(messages)  # can't call tools, must just answer
    return {"messages": [response]}

def agent_node(state: AgentState) -> AgentState:
    messages = [m for m in state["messages"]]
    
    models_to_try = [
        ("llama-3.1-8b-instant", 0),
        ("llama-3.3-70b-versatile", 0.3),
    ]
    
    for attempt, (model_name, temp) in enumerate(models_to_try):
        try:
            llm_attempt = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=temp
            ).bind_tools(rag_tools)
            
            response = llm_attempt.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"  ⚠️ LLM call failed (attempt {attempt + 1}, model={model_name}): {e}")
    
    return {"messages": [AIMessage(content="I'm having trouble processing that request. Could you rephrase your question?")]}

def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    db = state["db_session"]
    tool_messages = []
    terminal_used = False

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  🔧 Tool: {tool_name}({tool_args})")

        if tool_name == "search_documents":
            result = retrieve_with_confidence(db=db, query=tool_args["query"], confidence_threshold=0.8)
            if not result["confident"]:
                result_text = f"No relevant information found. Reason: {result['reason']}"
            else:
                chunks_text = "\n\n".join([
                    f"[Source: {c['metadata']['filename']}, chunk {c['metadata']['chunk_index']}]\n{c['content']}"
                    for c in result["chunks"]
                ])
                result_text = chunks_text

        elif tool_name == "list_available_documents":
            documents = db.query(Document).filter(Document.status == "ready").all()
            result_text = "No documents currently available." if not documents else "\n".join(
                [f"- {d.filename} ({d.chunk_count} chunks)" for d in documents]
            )

        elif tool_name == "summarize_document":
            result_text = summarize_document(db, tool_args["document_name"])
            if not result_text.startswith("No document found"):
                terminal_used = True  # ✅ mark success — force stop after this

        elif tool_name == "compare_documents":
            result_text = compare_documents(db, tool_args["document_a"], tool_args["document_b"])
            if "No document found" not in result_text:
                terminal_used = True  # ✅ mark success — force stop after this

        else:
            result_text = f"Unknown tool: {tool_name}"

        print(f"  👁️  Result preview: {result_text[:100]}...")

        tool_messages.append(ToolMessage(
            content=result_text,
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_messages, "terminal_tool_used": terminal_used}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def route_after_tools(state: AgentState) -> str:
    if state.get("terminal_tool_used"):
        return END  # skip agent_final entirely — tool output IS the final answer
    return "agent"

# ── Build Graph ──
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_conditional_edges("tools", route_after_tools, {END: END, "agent": "agent"})

rag_agent_app = graph.compile()

# ── Public function ──
def ask_agent(db: Session, question: str) -> dict:
    system_prompt = """You are DocuMind, an AI assistant that helps users find information in their uploaded documents.

Rules:
- ALWAYS use search_documents for specific questions about policies, procedures, or document content.
- Use summarize_document when the user wants an overview or summary of a whole document. Once you get a summary result, use it directly as your final answer — do NOT call search_documents afterward.
- Use compare_documents when the user wants to compare two documents. Once you get a comparison result, use it directly as your final answer.
- If search returns "No relevant information found", tell the user honestly — do not make up an answer.
- Always cite which document/source your answer comes from.
- For general greetings or questions unrelated to documents, respond naturally without searching.
- Use list_available_documents if the user asks what documents exist.
- IMPORTANT: Once a tool returns useful information that answers the user's question, generate your final answer immediately. Do not call additional unnecessary tools."""


    result = rag_agent_app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ],
        "db_session": db,
        "terminal_tool_used": False
    })

    final_message = result["messages"][-1]
    
    # If the graph ended right after a tool call, the last message IS the tool result
    if isinstance(final_message, ToolMessage):
        answer_text = final_message.content
    else:
        answer_text = final_message.content

    tools_used = []
    for m in result["messages"]:
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            tools_used.extend([tc["name"] for tc in m.tool_calls])

    return {
        "answer": answer_text,
        "tools_used": list(set(tools_used))
    }