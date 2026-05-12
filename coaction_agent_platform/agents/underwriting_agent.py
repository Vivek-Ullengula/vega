# coaction_agent_platform/agents/underwriting_agent.py
"""Strands-based underwriting agent — fully configurable via ExecutionProfile.

Ported from coactionbot/app/services/bedrock_kb_agent.py.
"""

import re
import asyncio
import structlog
from typing import AsyncGenerator

from strands import Agent
from strands.models.bedrock import BedrockModel

from coaction_agent_platform.domain.models import (
    ExecutionProfile,
    AgentInvocationResponse,
    SourceCitation,
)
from coaction_agent_platform.agents.prompts import get_prompt
from coaction_agent_platform.agents.tools.retriever import (
    search_manuals,
    configure_retriever,
    get_last_retrieval_sources,
)

logger = structlog.get_logger(__name__)


def _normalize_question(text: str) -> str:
    """Normalize question text for stable dedup checks."""
    if not text:
        return ""
    normalized = text.strip().lower()
    normalized = re.sub(r"^\d+\.\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized


def _extract_followups_from_text(content: str) -> list[str]:
    """Extract follow-up questions from assistant response text."""
    if not content:
        return []
    fu_pattern = r"(?i)\*{0,2}\s*You might also want to ask:?\s*\*{0,2}"
    if not re.search(fu_pattern, content):
        return []
    section = re.split(fu_pattern, content, maxsplit=1)[1]
    matches = re.findall(r"\d+\.\s*(.+)", section)
    return [m.strip() for m in matches if m.strip()]


class UnderwritingAgent:
    """Configurable Strands Agent for Coaction underwriting queries.

    Initialized from an ExecutionProfile, which determines:
    - Which Bedrock model to use
    - Which Knowledge Bases to query
    - What system prompt template to apply
    """

    def __init__(self, profile: ExecutionProfile, region: str = "us-east-1"):
        self.profile = profile
        self.region = region
        self._agents: dict[str, Agent] = {}  # keyed by role

        # Configure the retriever tool with KB IDs from the profile
        configure_retriever(
            knowledge_base_ids=profile.retrieval_profile.knowledge_base_ids,
            region=region,
        )

        logger.info(
            "underwriting_agent_initialized",
            agent_id=profile.agent_id,
            model=profile.model_profile.model_id,
            kb_ids=profile.retrieval_profile.knowledge_base_ids,
        )

    def _build_agent(self, role: str) -> Agent:
        """Build a Strands Agent for the given user role."""
        mp = self.profile.model_profile

        model = BedrockModel(
            model_id=mp.model_id,
            region_name=self.region,
            temperature=mp.temperature,
            max_tokens=mp.max_tokens or 4096,
        )

        prompt = get_prompt(self.profile.prompt_template_id, role)

        return Agent(
            model=model,
            system_prompt=prompt,
            tools=[search_manuals],
        )

    def _get_or_create_agent(self, role: str) -> Agent:
        """Get or create a cached agent for the given role."""
        role_key = (role or "").strip().lower()
        if role_key not in self._agents:
            self._agents[role_key] = self._build_agent(role_key)
        return self._agents[role_key]

    async def invoke(
        self,
        query: str,
        role: str = "agent",
        history: list[dict] | None = None,
    ) -> dict:
        """Invoke the agent with a query.

        Args:
            query: The user's question.
            role: User role (agent/underwriter/external).
            history: Previous conversation messages for context.

        Returns:
            dict with 'answer', 'citations', 'follow_up_questions', 'sources'.
        """
        agent = self._get_or_create_agent(role)

        # Restore conversation history if provided
        if history:
            agent.state.messages = history.copy()

        # Execute the agent (synchronous Strands call)
        response = agent(query)
        answer = str(response)

        # Extract follow-up questions from the answer
        follow_up_questions = []
        fu_pattern = r"(?i)\*{0,2}\s*You might also want to ask:?\s*\*{0,2}"
        if re.search(fu_pattern, answer):
            parts = re.split(fu_pattern, answer, maxsplit=1)
            clean_answer = parts[0].strip()
            fu_text = parts[1]
            matches = re.findall(r"\d+\.\s*(.+)", fu_text)
            raw_followups = [m.strip() for m in matches if m.strip()]

            # Dedup against history
            historical_questions: set[str] = set()
            if history:
                for msg in history:
                    msg_role = (msg.get("role") or "").strip().lower()
                    content = msg.get("content") or ""
                    if msg_role == "user":
                        nq = _normalize_question(content)
                        if nq:
                            historical_questions.add(nq)
                    elif msg_role == "assistant":
                        for prev_fu in _extract_followups_from_text(content):
                            nfu = _normalize_question(prev_fu)
                            if nfu:
                                historical_questions.add(nfu)

            seen: set[str] = set()
            for question in raw_followups:
                normalized = _normalize_question(question)
                if not normalized or normalized in historical_questions or normalized in seen:
                    continue
                seen.add(normalized)
                follow_up_questions.append(question)
                if len(follow_up_questions) == 3:
                    break

            answer = clean_answer

        # Get source citations
        retrieval_sources = get_last_retrieval_sources()
        all_urls = [s["url"] for s in retrieval_sources if s.get("url") and s["url"] != "N/A"]
        cited_urls = [url for url in all_urls if url in answer]
        sources = cited_urls if cited_urls else all_urls[:3]

        # Build citation objects
        citations = []
        seen_urls = set()
        for s in retrieval_sources:
            url = s.get("url", "")
            if url in sources and url not in seen_urls:
                seen_urls.add(url)
                citations.append(
                    SourceCitation(
                        source_id=url,
                        title=s.get("heading", "") or url,
                        uri=url,
                        manual_name=s.get("manual_name", ""),
                    )
                )

        # Get the current agent messages for session persistence
        current_messages = agent.state.messages if hasattr(agent.state, "messages") else []

        return {
            "answer": answer,
            "citations": citations,
            "follow_up_questions": follow_up_questions,
            "sources": sources,
            "agent_messages": current_messages,
        }
