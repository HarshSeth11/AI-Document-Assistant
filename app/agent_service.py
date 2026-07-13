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
    }
]

llm_with_tools = llm.bind_tools(rag_tools)

# ── State — carries db session through the graph ──
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    db_session: Session  # not serialized, just passed through

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

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  🔧 Tool: {tool_name}({tool_args})")

        if tool_name == "search_documents":
            result = retrieve_with_confidence(
                db=db,
                query=tool_args["query"],
                confidence_threshold=0.8
            )
            
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
            if not documents:
                result_text = "No documents currently available."
            else:
                result_text = "\n".join([f"- {d.filename} ({d.chunk_count} chunks)" for d in documents])

        else:
            result_text = f"Unknown tool: {tool_name}"

        print(f"  👁️  Result preview: {result_text[:100]}...")

        tool_messages.append(ToolMessage(
            content=result_text,
            tool_call_id=tool_call["id"]
        ))

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ── Build Graph ──
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
rag_agent_app = graph.compile()

# ── Public function ──
def ask_agent(db: Session, question: str) -> dict:
    system_prompt = """You are DocuMind, an AI assistant that helps users find information in their uploaded documents.

Rules:
- ALWAYS use search_documents for questions about policies, procedures, or document content.
- If search returns "No relevant information found", tell the user honestly — do not make up an answer.
- Always cite which document/source your answer comes from.
- For general greetings or questions unrelated to documents, respond naturally without searching.
- Use list_available_documents if the user asks what documents exist.
- When calling a tool, use proper function calling — never write out function calls as text."""

    result = rag_agent_app.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ],
        "db_session": db
    })

    final_message = result["messages"][-1]
    
    # Track which tools were used
    tools_used = []
    for m in result["messages"]:
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            tools_used.extend([tc["name"] for tc in m.tool_calls])

    return {
        "answer": final_message.content,
        "tools_used": list(set(tools_used))
    }