"""Conversation history and ticket-triage chains."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from prompts import (
    CLASSIFICATION_PROMPT,
    DETAILS_PROMPT,
    DRAFT_REPLY_PROMPT,
    PRIORITY_PROMPT,
    SUPPORT_PROMPT,
)

store: dict[str, InMemoryChatMessageHistory] = {}

SAMPLE_TICKETS = {
    "Possible defect - peeling sole": (
        "My son's Dino-Print Sneakers (INR 2,499) were delivered 14 days ago. "
        "The sole is peeling after 3 wears. I can send photos."
    ),
    "Standard return - unworn": (
        "I ordered a jacket for INR 1,800, delivered 10 days ago. It is unworn "
        "with tags and in its original packaging, but the color doesn't suit me."
    ),
    "Outside standard return window": (
        "I want to return a pair of shoes because I changed my mind. They were "
        "delivered 45 days ago and have been worn."
    ),
    "Mandatory human review - allergy": (
        "My child developed a rash after wearing the pajamas. The order was "
        "INR 1,200 and arrived yesterday."
    ),
}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the shared in-memory history for a conversation."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def build_chatbot(llm: object) -> RunnableWithMessageHistory:
    """Build a chatbot that reads and updates per-session message history."""
    return RunnableWithMessageHistory(
        SUPPORT_PROMPT | llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def run_triage_chain(ticket: str, llm: object) -> dict[str, str]:
    """Run classification, extraction, prioritization, and reply drafting."""
    parser = StrOutputParser()
    category = (CLASSIFICATION_PROMPT | llm | parser).invoke(
        {"ticket_text": ticket}
    ).strip()
    details = (DETAILS_PROMPT | llm | parser).invoke({"ticket_text": ticket}).strip()
    priority = (PRIORITY_PROMPT | llm | parser).invoke(
        {"ticket_text": ticket, "category": category, "details": details}
    ).strip()
    draft_reply = (DRAFT_REPLY_PROMPT | llm | parser).invoke(
        {
            "ticket_text": ticket,
            "category": category,
            "details": details,
            "priority": priority,
        }
    ).strip()
    return {
        "category": category,
        "details": details,
        "priority": priority,
        "draft_reply": draft_reply,
    }
