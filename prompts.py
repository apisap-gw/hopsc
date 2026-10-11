"""Prompt templates for Hopscotch customer support and ticket triage."""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

POLICY = (Path(__file__).with_name("hopscotch_policy.md")).read_text(encoding="utf-8")

SUPPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful Hopscotch customer-support assistant.
Answer using only the policy below. Be empathetic, concise, and ask for missing
information when needed. Never promise an outcome that the policy does not allow.
Always refer child-safety concerns, skin reactions, orders over INR 5,000, and
unclear or conflicting cases to a human agent; do not make those decisions
automatically.

Policy:
{policy}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
).partial(policy=POLICY)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the customer ticket using the Hopscotch policy.
Return exactly one category: policy_violation, genuine_defect, or other.
Use policy_violation for requests that the policy says are not covered; use
genuine_defect for a possible manufacturing defect; otherwise use other.
Return only the category, with no explanation.

Policy:
{policy}""",
        ),
        ("human", "{ticket_text}"),
    ]
).partial(policy=POLICY)

DETAILS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract details stated in the ticket. Return only a JSON object with
these keys: customer_name, order_id, product, issue, days_since_delivery,
order_value_inr, evidence, and human_review_required. Use null for unknown
values. Set human_review_required to true if the case involves child safety,
skin reaction/allergy, an order over INR 5,000, or unclear/conflicting facts.
Do not infer missing facts.""",
        ),
        ("human", "{ticket_text}"),
    ]
)

PRIORITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Assign a ticket priority of high, medium, or low. Use high for
safety/allergy risks, imminent harm, or an order requiring mandatory human
review; medium for a plausible defect or time-sensitive issue; otherwise low.
Return only the priority, with no explanation.""",
        ),
        (
            "human",
            "Ticket:\n{ticket_text}\n\nClassification: {category}\n\nExtracted details: {details}",
        ),
    ]
)

DRAFT_REPLY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Draft a concise, empathetic customer reply using the policy below.
Do not claim a refund or replacement has been approved. Ask for photo evidence
for possible manufacturing defects. If human review is required, say that a
human agent will review the case and do not decide it automatically.

Policy:
{policy}""",
        ),
        (
            "human",
            "Ticket:\n{ticket_text}\n\nClassification: {category}\n\nExtracted details: {details}\n\nPriority: {priority}",
        ),
    ]
).partial(policy=POLICY)

PROMPT_CATALOG = {
    "Customer support": {
        "prompt": SUPPORT_PROMPT,
        "description": "Answer a customer while using conversation history and the support policy.",
        "used_in": "Chat tab",
    },
    "Ticket classification": {
        "prompt": CLASSIFICATION_PROMPT,
        "description": "Classify a ticket as a policy violation, possible defect, or other.",
        "used_in": "Ticket triage, step 1",
    },
    "Ticket detail extraction": {
        "prompt": DETAILS_PROMPT,
        "description": "Extract the stated ticket details as JSON without guessing.",
        "used_in": "Ticket triage, step 2",
    },
    "Ticket priority": {
        "prompt": PRIORITY_PROMPT,
        "description": "Assign a priority using the ticket and extracted information.",
        "used_in": "Ticket triage, step 3",
    },
    "Draft customer reply": {
        "prompt": DRAFT_REPLY_PROMPT,
        "description": "Draft a policy-grounded reply, escalating mandatory human-review cases.",
        "used_in": "Ticket triage, step 4",
    },
}
