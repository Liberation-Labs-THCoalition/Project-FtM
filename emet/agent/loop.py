"""Agent loop — the investigate-reason-act cycle.

The core agentic runtime. Takes an investigation goal, iteratively
uses tools, accumulates findings, follows leads, and decides when
to stop.

Design:
  - Simple loop, not a DAG or planning framework
  - LLM decides next action from investigation context
  - One tool call per turn
  - Stops on budget exhaustion or LLM "conclude" signal
  - Falls back to heuristic routing if no LLM available

    agent = InvestigationAgent()
    result = await agent.investigate("Acme Corp corruption in Panama")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emet.agent.session import Session, Finding, Lead
from emet.agent.safety_harness import SafetyHarness
from emet.mcp.tools import EmetToolExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Available tools the agent can call
# ---------------------------------------------------------------------------

AGENT_TOOLS = {
    "search_entities": {
        "description": "Search for entities (people, companies, sanctions) across multiple sources",
        "params": ["query", "entity_type", "sources"],
    },
    "search_aleph": {
        "description": "Search Aleph (OCCRP investigative data) for entities, documents, leaked records",
        "params": ["query", "schema", "collection_ids", "countries"],
    },
    "osint_recon": {
        "description": "OSINT reconnaissance on a target (domain, email, IP)",
        "params": ["target", "scan_type"],
    },
    "analyze_graph": {
        "description": "Analyze network graph for hidden connections, brokers, communities",
        "params": ["entities", "analysis_type"],
    },
    "trace_ownership": {
        "description": "Trace corporate ownership chains",
        "params": ["entity_name", "max_depth"],
    },
    "screen_sanctions": {
        "description": "Screen entity against sanctions and PEP lists",
        "params": ["entity_name", "entity_type", "threshold"],
    },
    "investigate_blockchain": {
        "description": "Investigate blockchain address transactions and flows",
        "params": ["address", "chain"],
    },
    "monitor_entity": {
        "description": "Monitor real-time news for an entity via GDELT",
        "params": ["entity_name", "timespan"],
    },
    "generate_report": {
        "description": "Generate investigation report from accumulated findings",
        "params": ["title", "format"],
    },
    "conclude": {
        "description": "End the investigation and compile results",
        "params": [],
    },
}


# ---------------------------------------------------------------------------
# System prompt for LLM-powered decision making
# ---------------------------------------------------------------------------

INVESTIGATION_SYSTEM_PROMPT = """You are an investigative journalist's AI research assistant. Your job is to \
direct an investigation by choosing which tools to call next, based on the current state of evidence.

PRINCIPLES:
- Follow the money. Corporate ownership chains and financial flows reveal hidden connections.
- Verify through multiple sources. One data point is a hint; corroboration is evidence.
- Pursue the highest-value leads first. Sanctions hits and ownership anomalies outrank general searches.
- Know when to stop. Conclude when findings answer the goal or when remaining leads are low-priority.
- Never fabricate. If a tool returns no data, note the gap — don't invent results.

STRATEGY:
1. Broad entity search first to identify key players
2. Sanctions/PEP screening for flagged entities
3. Ownership tracing for corporate structures
4. OSINT for digital footprints when warranted
5. Blockchain investigation only when crypto addresses surface
6. Conclude and synthesize when the picture is clear

You respond with ONLY a single JSON object representing the next action. No explanations outside the JSON."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Agent configuration."""
    max_turns: int = 15            # Budget per investigation
    min_confidence: float = 0.3    # Minimum to pursue a lead
    auto_sanctions_screen: bool = True  # Always screen found entities
    auto_news_check: bool = True   # Always check GDELT for targets
    llm_provider: str = "stub"     # LLM backend (stub, ollama, anthropic)
    verbose: bool = True           # Log reasoning
    # Safety
    enable_safety: bool = True     # Enable full safety harness
    enable_pii_redaction: bool = True   # Scrub PII from outputs
    enable_shield: bool = True     # Budget/rate/circuit breaker
    tool_timeout_seconds: float = 60.0  # Per-tool-call timeout (prevents hangs)
    # Persistence
    persist_path: str = ""         # Auto-save session to this path
    memory_dir: str = ""           # Directory for cross-session memory (auto-recall prior findings)
    # Visualization
    generate_graph: bool = True    # Generate Cytoscape graph at conclusion
    # Demo
    demo_mode: bool = False        # Use bundled demo data when sources return empty
    # Output
    auto_pdf: bool = True          # Auto-generate PDF report after investigation
    output_dir: str = ""           # Directory for PDF/export output (default: memory_dir or cwd)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class InvestigationAgent:
    """The agentic investigator.

    Takes a goal, iteratively uses tools, and accumulates findings
    into a coherent investigation session.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        self._executor = EmetToolExecutor(demo_mode=self._config.demo_mode)
        # Safety harness
        if self._config.enable_safety:
            self._harness = SafetyHarness.from_defaults()
        else:
            self._harness = SafetyHarness.disabled()
        # LLM client — created once, reused across turns
        self._llm_client: Any = None  # Lazy-initialized
        self._cost_tracker: Any = None
        # Audit archive — forensic record of every tool call, LLM exchange, CoT
        self._audit: Any = None  # Created per-investigation in investigate()

    async def investigate(self, goal: str) -> Session:
        """Run an investigation from a natural-language goal.

        Returns the completed Session with all findings, entities,
        leads, and reasoning trace.
        """
        session = Session(goal=goal)
        session.record_reasoning(f"Starting investigation: {goal}")

        # Open audit archive for forensic recording
        from emet.agent.audit import AuditArchive
        audit_dir = self._config.memory_dir or "investigations/audit"
        self._audit = AuditArchive(base_dir=Path(audit_dir) / "audit")
        self._audit.open(session.id, goal=goal)

        # Phase 0: Recall prior intelligence from memory
        if self._config.memory_dir:
            prior = self._recall_prior_intelligence(session)
            if prior:
                session.record_reasoning(
                    f"Memory recall: {len(prior)} relevant findings from prior investigations"
                )
                session._prior_intelligence = prior

        # Phase 1: Initial search
        await self._initial_search(session)

        # Phase 2: Iterative investigation loop
        while session.turn_count < self._config.max_turns:
            session.turn_count += 1

            action = await self._decide_next_action(session)

            if action["tool"] == "conclude":
                session.record_reasoning("Concluding investigation.")
                break

            source = action.get("_decision_source", "?")
            session.record_reasoning(
                f"Turn {session.turn_count} [{source}]: {action['reasoning']}"
            )

            # --- Safety: pre-check ---
            verdict = self._harness.pre_check(
                tool=action["tool"],
                args=action.get("args", {}),
            )
            if verdict.blocked:
                session.record_reasoning(
                    f"BLOCKED by safety harness: {verdict.reason}"
                )
                if self._audit:
                    self._audit.record_safety(
                        check_type="pre_check",
                        tool=action["tool"],
                        verdict="blocked",
                        details={"reason": verdict.reason, "args": action.get("args", {})},
                    )
                # Mark lead as dead end if it was lead-following
                lead_id = action.get("lead_id")
                if lead_id:
                    session.resolve_lead(lead_id, "blocked")
                continue

            result = await self._execute_action(session, action)

            await self._process_result(session, action, result)

            # Check if we've exhausted leads
            if not session.get_open_leads() and session.turn_count >= 3:
                session.record_reasoning(
                    "No open leads remaining. Concluding."
                )
                break

        # Phase 3: Generate report
        await self._generate_report(session)

        # Phase 4: Generate investigation graph
        if self._config.generate_graph:
            self._generate_investigation_graph(session)

        # Phase 5: Persist session
        if self._config.persist_path:
            from emet.agent.persistence import save_session
            save_session(session, self._config.persist_path)

        # Auto-save to memory directory for cross-session recall
        if self._config.memory_dir:
            from emet.agent.persistence import save_session
            mem_path = Path(self._config.memory_dir) / f"{session.id}.json"
            try:
                save_session(session, mem_path)
                session.record_reasoning(f"Session saved to memory: {mem_path}")
            except Exception as exc:
                logger.debug("Memory auto-save failed: %s", exc)

        # Attach safety audit to session
        session._safety_audit = self._harness.audit_summary()

        # Close audit archive — compress, hash, write to disk
        if self._audit:
            try:
                # Record all reasoning steps into the archive
                for thought in session.reasoning_trace:
                    self._audit.record_reasoning(thought)
                # Record all findings
                for finding in session.findings:
                    self._audit.record_event("finding", {
                        "id": finding.id,
                        "source": finding.source,
                        "summary": finding.summary,
                        "confidence": finding.confidence,
                        "entity_count": len(finding.entities),
                        "entities": finding.entities,
                    })
                # Record all leads
                for lead in session.leads:
                    self._audit.record_event("lead", {
                        "id": lead.id,
                        "description": lead.description,
                        "priority": lead.priority,
                        "tool": lead.tool,
                        "query": lead.query,
                        "status": lead.status,
                    })
                manifest = self._audit.close(final_summary=session.summary())
                session._audit_manifest = {
                    "path": manifest.path,
                    "sha256": manifest.sha256,
                    "event_count": manifest.event_count,
                    "compressed_bytes": manifest.compressed_bytes,
                    "uncompressed_bytes": manifest.uncompressed_bytes,
                }
            except Exception as exc:
                logger.debug("Audit archive close failed: %s", exc)

        # Post-investigation: suggest next steps
        try:
            from emet.agent.health import suggest_next_steps
            summary = session.summary()
            next_steps = suggest_next_steps(summary)
            if next_steps:
                session._next_steps = [
                    {"action": s.action, "reason": s.reason, "command": s.command}
                    for s in next_steps
                ]
        except Exception:
            pass  # Non-critical

        return session

    async def _initial_search(self, session: Session) -> None:
        """Phase 1: Cast the net — search entities, check news."""
        goal = session.goal

        # Entity search
        session.record_reasoning(f"Initial entity search for: {goal}")
        try:
            result = await asyncio.wait_for(
                self._executor.execute_raw(
                    "search_entities",
                    {"query": goal, "entity_type": "Any", "limit": 20},
                ),
                timeout=self._config.tool_timeout_seconds,
            )
            session.record_tool_use("search_entities", {"query": goal}, result)

            # Audit: capture initial search with full result
            if self._audit:
                self._audit.record_tool_call(
                    tool="search_entities",
                    args={"query": goal, "entity_type": "Any", "limit": 20},
                    result=result,
                    decision_source="initial_phase",
                )

            entities = result.get("entities", [])

            # Demo mode: always inject bundled demo data for a compelling
            # first-use experience (real results may be sparse or irrelevant)
            if self._config.demo_mode:
                from emet.data.demo_entities import get_demo_entities
                demo_entities = get_demo_entities(goal)
                if demo_entities:
                    # Use demo entities as the primary dataset; they form a
                    # coherent investigation network that demonstrates all
                    # pipeline capabilities
                    entities = demo_entities
                    session.record_reasoning(
                        f"Demo mode: loaded {len(entities)} entities "
                        f"(Meridian Holdings investigation scenario)"
                    )
                    result["entities"] = entities
                    result["source_stats"] = {"demo_dataset": len(entities)}
                    result["result_count"] = len(entities)

            if entities:
                finding = Finding(
                    source="search_entities",
                    summary=f"Found {len(entities)} entities matching '{goal}'",
                    entities=entities,
                    confidence=0.7,
                )
                session.add_finding(finding)

                # Generate leads from found entities
                for entity in entities[:5]:
                    props = entity.get("properties", {})
                    names = props.get("name", [])
                    schema = entity.get("schema", "")
                    if names:
                        name = names[0]
                        # Sanctions screening lead
                        if self._config.auto_sanctions_screen:
                            session.add_lead(Lead(
                                description=f"Screen {name} against sanctions",
                                priority=0.8,
                                source_finding=finding.id,
                                query=name,
                                tool="screen_sanctions",
                            ))
                        # Ownership tracing for companies
                        if schema in ("Company", "Organization", "LegalEntity"):
                            session.add_lead(Lead(
                                description=f"Trace ownership of {name}",
                                priority=0.7,
                                source_finding=finding.id,
                                query=name,
                                tool="trace_ownership",
                            ))

        except Exception as exc:
            session.record_reasoning(f"Initial search failed: {exc}")

        # News monitoring
        if self._config.auto_news_check:
            try:
                result = await asyncio.wait_for(
                    self._executor.execute_raw(
                        "monitor_entity",
                        {"entity_name": goal, "timespan": "7d"},
                    ),
                    timeout=self._config.tool_timeout_seconds,
                )
                session.record_tool_use("monitor_entity", {"entity_name": goal}, result)

                # Audit: capture news monitoring with full result
                if self._audit:
                    self._audit.record_tool_call(
                        tool="monitor_entity",
                        args={"entity_name": goal, "timespan": "7d"},
                        result=result,
                        decision_source="initial_phase",
                    )

                articles = result.get("article_count", 0)
                if articles:
                    session.add_finding(Finding(
                        source="monitor_entity",
                        summary=f"Found {articles} recent news articles about '{goal}'",
                        confidence=0.5,
                        raw_data=result,
                    ))
            except Exception as exc:
                session.record_reasoning(f"News check failed: {exc}")

    async def _decide_next_action(
        self, session: Session
    ) -> dict[str, Any]:
        """Decide what to do next.

        Uses LLM if available, falls back to lead-following heuristic.
        Rejects duplicate tool calls (same tool + same key args).
        """
        # Try LLM decision
        action = await self._llm_decide(session)
        if action and not self._is_duplicate(session, action):
            action["_decision_source"] = "llm"
            return action
        elif action:
            session.record_reasoning(
                f"LLM suggested duplicate call: {action['tool']}. Falling back."
            )

        # Heuristic fallback: follow highest-priority open lead
        if self._config.llm_provider != "stub":
            session.record_reasoning(
                "LLM decision unavailable — falling back to heuristic lead-following"
            )
        action = self._heuristic_decide(session)
        if self._is_duplicate(session, action):
            session.record_reasoning(
                f"Heuristic also duplicate: {action['tool']}. Concluding."
            )
            return {"tool": "conclude", "args": {}, "reasoning": "All actions duplicate previous calls"}
        action["_decision_source"] = "heuristic"
        return action

    @staticmethod
    def _is_duplicate(session: "Session", action: dict[str, Any]) -> bool:
        """Check if this tool call duplicates a previous one."""
        tool = action.get("tool", "")
        if tool in ("conclude", "generate_report"):
            return False  # Always allow terminal actions

        args = action.get("args", {})
        # Build a comparable key from tool + significant args
        key_args = {
            k: v for k, v in sorted(args.items())
            if k not in ("entities", "params", "format", "include_graph",
                         "include_timeline", "entity_ids")
            and v  # skip empty/falsy
        }
        sig = (tool, tuple(key_args.items()))

        for prev in session.tool_history:
            prev_args = prev.get("args", {})
            prev_key = {
                k: v for k, v in sorted(prev_args.items())
                if k not in ("entities", "params", "format", "include_graph",
                             "include_timeline", "entity_ids")
                and v
            }
            prev_sig = (prev.get("tool", ""), tuple(prev_key.items()))
            if sig == prev_sig:
                return True
        return False

    def _get_llm_client(self) -> Any:
        """Get or create the cached LLM client.

        Returns None if the provider is unavailable (e.g. no API key).
        The client is created once and reused for the duration of the
        investigation, maintaining cost tracking across turns.
        """
        if self._llm_client is not None:
            return self._llm_client

        try:
            from emet.cognition.llm_factory import create_llm_client_sync
            from emet.cognition.model_router import CostTracker

            self._cost_tracker = CostTracker()
            self._llm_client = create_llm_client_sync(
                provider=self._config.llm_provider,
                cost_tracker=self._cost_tracker,
            )
            return self._llm_client
        except ImportError as exc:
            logger.warning(
                "LLM unavailable — missing dependency: %s "
                "(install with: pip install %s)",
                exc,
                str(exc).split("'")[-2] if "'" in str(exc) else "missing-package",
            )
            return None
        except Exception as exc:
            logger.debug("Cannot create LLM client: %s", exc)
            return None

    async def _llm_decide(self, session: Session) -> dict[str, Any] | None:
        """Ask the LLM what to do next.

        Uses the investigation context and available tools to prompt the
        LLM for a structured JSON action. Returns None if the LLM is
        unavailable or the response can't be parsed, triggering heuristic
        fallback.
        """
        client = self._get_llm_client()
        if client is None:
            return None

        context = session.context_for_llm(max_turns=self._config.max_turns)
        tools_desc = "\n".join(
            f"  - {name}: {info['description']}"
            f"\n    Parameters: {', '.join(info['params'])}"
            for name, info in AGENT_TOOLS.items()
        )

        system = INVESTIGATION_SYSTEM_PROMPT

        prompt = f"""{context}

AVAILABLE TOOLS:
{tools_desc}

Based on the investigation state above, decide the SINGLE next action.
Respond with ONLY a JSON object — no markdown, no commentary:
{{"tool": "<tool_name>", "args": {{<relevant_params>}}, "reasoning": "<one sentence why>"}}

Rules:
- Pick the tool that fills the biggest gap in your current knowledge
- If you have open leads, consider following the highest-priority one
- If findings are sufficient to answer the goal, use "conclude"
- Don't repeat the same tool call with the same arguments
- Each tool call costs budget — be efficient"""

        try:
            response = await client.complete(
                prompt,
                system=system,
                max_tokens=300,
                temperature=0.2,  # Low temp for structured decisions
                tier="balanced",
            )
            text = response.text.strip()

            # Parse JSON from response (handles preamble/markdown fences)
            if "```" in text:
                # Strip markdown code fences
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                action = json.loads(json_str)

                # Validate tool name
                tool = action.get("tool", "")
                if tool not in AGENT_TOOLS:
                    logger.debug("LLM suggested unknown tool %r", tool)
                    return None

                # Ensure args is a dict
                if not isinstance(action.get("args"), dict):
                    action["args"] = {}

                logger.info(
                    "LLM decision (turn %d): %s — %s",
                    session.turn_count,
                    action["tool"],
                    action.get("reasoning", "")[:80],
                )

                # Audit: record full LLM exchange
                if self._audit:
                    self._audit.record_llm_exchange(
                        system_prompt=system,
                        user_prompt=prompt,
                        raw_response=response.text,
                        parsed_action=action,
                        model=getattr(client, "model", self._config.llm_provider),
                    )

                return action

        except ImportError as exc:
            logger.warning("LLM decision failed (missing dependency): %s", exc)
        except Exception as exc:
            logger.warning("LLM decision failed: %s", exc)
            # Audit: record failed LLM exchange
            if self._audit:
                self._audit.record_event("llm_error", {
                    "error": str(exc),
                    "system_prompt": system,
                    "user_prompt": prompt,
                })

        return None

    def _heuristic_decide(self, session: Session) -> dict[str, Any]:
        """Fallback: phase-aware heuristic investigation strategy.

        Phases:
          1. EXPLORE — follow high-priority leads (sanctions, ownership)
          2. ANALYZE — run graph analysis once enough entities accumulated
          3. CONCLUDE — generate report and end
        """
        leads = session.get_open_leads()
        turn = session.turn_count
        max_turns = self._config.max_turns
        remaining = max_turns - turn

        # --- Phase 3: Running out of budget → wrap up ---
        if remaining <= 1:
            return {
                "tool": "conclude",
                "args": {},
                "reasoning": "Budget exhausted — concluding investigation",
            }

        # --- Phase 2: Enough entities → graph analysis (run once) ---
        already_analyzed = any(
            t.get("tool") == "analyze_graph" for t in session.tool_history
        )
        if (
            not already_analyzed
            and session.entity_count >= 8
            and (remaining <= 3 or turn >= max_turns // 2)
        ):
            return {
                "tool": "analyze_graph",
                "args": {"algorithm": "full"},
                "reasoning": f"Accumulated {session.entity_count} entities — running network analysis before concluding",
            }

        # --- Phase 1: Follow leads ---
        if not leads:
            return {"tool": "conclude", "args": {}, "reasoning": "No leads remaining"}

        lead = leads[0]
        lead.status = "investigating"

        args: dict[str, Any] = {}
        if lead.tool == "screen_sanctions":
            args = {
                "entity_name": lead.query,
                "entity_type": "LegalEntity",
                "threshold": 0.6,
            }
        elif lead.tool == "trace_ownership":
            args = {"entity_name": lead.query, "max_depth": 3}
        elif lead.tool == "osint_recon":
            args = {"target": lead.query, "scan_type": "passive"}
        elif lead.tool == "search_entities":
            args = {"query": lead.query, "entity_type": "Any"}
        elif lead.tool == "investigate_blockchain":
            args = {"address": lead.query, "chain": "ethereum"}
        elif lead.tool == "monitor_entity":
            args = {"entity_name": lead.query, "timespan": "24h"}
        elif lead.tool == "analyze_graph":
            args = {"algorithm": "full"}
        else:
            args = {"query": lead.query}

        return {
            "tool": lead.tool or "search_entities",
            "args": args,
            "reasoning": f"Following lead: {lead.description}",
            "lead_id": lead.id,
        }

    async def _execute_action(
        self,
        session: Session,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool call with safety observation and audit recording."""
        tool = action["tool"]
        args = action.get("args", {})
        import time
        t0 = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._executor.execute_raw(tool, args),
                timeout=self._config.tool_timeout_seconds,
            )
            duration_ms = (time.monotonic() - t0) * 1000

            # Safety: report success to circuit breaker
            self._harness.report_tool_success(tool)

            # Safety: observe result (audit-only, no scrubbing)
            result_text = json.dumps(result, default=str)
            if len(result_text) > 10:
                self._harness.post_check(result_text, tool=tool)

            session.record_tool_use(tool, args, result)

            # Audit: record complete tool call with full result
            if self._audit:
                self._audit.record_tool_call(
                    tool=tool,
                    args=args,
                    result=result,
                    duration_ms=duration_ms,
                    decision_source=action.get("_decision_source", ""),
                )

            return result
        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000

            # Safety: report failure to circuit breaker
            self._harness.report_tool_failure(tool)

            error = {"error": str(exc), "tool": tool}
            session.record_tool_use(tool, args, error)
            session.record_reasoning(f"Tool {tool} failed: {exc}")

            # Audit: record failed tool call
            if self._audit:
                self._audit.record_tool_call(
                    tool=tool,
                    args=args,
                    result=error,
                    duration_ms=duration_ms,
                    decision_source=action.get("_decision_source", ""),
                )

            # Mark lead as dead end if it was a lead-following action
            lead_id = action.get("lead_id")
            if lead_id:
                session.resolve_lead(lead_id, "dead_end")
            return error

    async def _process_result(
        self,
        session: Session,
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Process tool result into findings and new leads."""
        tool = action["tool"]
        lead_id = action.get("lead_id")

        if "error" in result:
            return

        # Extract entities from tool-specific locations.
        # Different tools return data under different keys:
        #   search_entities, trace_ownership, monitor_entity → "entities"
        #   screen_sanctions → "matches" (no "entities" key)
        #   investigate_blockchain → "data" (nested dict)
        #   analyze_graph → "result" (algorithm output)
        #   check_alerts → "alerts" (change detection)
        #   ingest_documents → "documents" (extracted docs)
        #   generate_report → "report" (terminal, no entities)
        entities = result.get("entities", [])
        matches = result.get("matches", [])

        # Determine if this result has meaningful data worth recording.
        # Check all known data-carrying keys to avoid silent drops.
        result_count = result.get(
            "result_count",
            result.get(
                "match_count",
                result.get(
                    "alert_count",
                    result.get("document_count", len(entities)),
                ),
            ),
        )
        has_data = bool(
            entities
            or result_count
            or matches
            or result.get("data")
            or result.get("report")
            or result.get("result")      # analyze_graph
            or result.get("alerts")      # check_alerts
            or result.get("documents")   # ingest_documents
        )

        if has_data:
            finding = Finding(
                source=tool,
                summary=_build_finding_summary(tool, action, result),
                entities=entities,
                confidence=_estimate_confidence(result),
                raw_data={k: v for k, v in result.items() if k != "entities"},
            )
            session.add_finding(finding)

            # Generate new leads from results
            self._extract_leads(session, finding, result)

        # Resolve the lead that triggered this action
        if lead_id:
            session.resolve_lead(lead_id, "resolved")

    def _extract_leads(
        self,
        session: Session,
        finding: Finding,
        result: dict[str, Any],
    ) -> None:
        """Extract new investigative leads from a finding."""
        for entity in finding.entities[:3]:
            props = entity.get("properties", {})
            schema = entity.get("schema", "")
            names = props.get("name", [])
            if not names:
                continue
            name = names[0]

            # Don't create duplicate leads
            existing_queries = {l.query.lower() for l in session.leads}
            if name.lower() in existing_queries:
                continue

            # Companies → trace ownership
            if schema in ("Company", "Organization", "LegalEntity"):
                session.add_lead(Lead(
                    description=f"Trace ownership of {name}",
                    priority=0.6,
                    source_finding=finding.id,
                    query=name,
                    tool="trace_ownership",
                ))

            # Crypto addresses → investigate blockchain
            crypto_keys = props.get("publicKey", [])
            for addr in crypto_keys[:1]:
                session.add_lead(Lead(
                    description=f"Investigate crypto address {addr[:20]}...",
                    priority=0.5,
                    source_finding=finding.id,
                    query=addr,
                    tool="investigate_blockchain",
                ))

        # Sanctions matches → high priority
        matches = result.get("matches", [])
        for match in matches[:3]:
            match_name = match.get("name", match.get("entity_name", ""))
            if match_name:
                session.add_lead(Lead(
                    description=f"SANCTIONS HIT: {match_name}",
                    priority=0.95,
                    source_finding=finding.id,
                    query=match_name,
                    tool="search_entities",
                ))

    async def _generate_report(self, session: Session) -> None:
        """Generate final investigation report.

        This is a publication boundary — PII is scrubbed from the
        report output before it's attached to the session.

        When an LLM is available, it synthesizes findings into a
        coherent narrative. Otherwise falls back to the template-based
        generate_report tool.
        """
        try:
            # Try LLM-powered synthesis first
            synthesized = await self._llm_synthesize_report(session)
            if synthesized:
                result = {
                    "report": synthesized,
                    "format": "markdown",
                    "source": "llm_synthesis",
                }
            else:
                # Fallback: template-based report from tool
                entity_summaries = []
                for eid, entity in list(session.entities.items())[:50]:
                    names = entity.get("properties", {}).get("name", [])
                    schema = entity.get("schema", "")
                    entity_summaries.append(f"[{schema}] {names[0] if names else eid}")

                result = await asyncio.wait_for(
                    self._executor.execute_raw(
                        "generate_report",
                        {
                            "title": f"Investigation: {session.goal}",
                            "format": "markdown",
                            "entities": list(session.entities.values())[:50],
                        },
                    ),
                    timeout=self._config.tool_timeout_seconds,
                )

            # Publication boundary: scrub PII from report output
            if self._config.enable_pii_redaction:
                result = self._harness.scrub_dict_for_publication(result, "report")

            # Store report on session for programmatic access
            session.report = result.get("report", "")

            session.record_tool_use("generate_report", {"title": session.goal}, result)
            session.record_reasoning("Report generated (PII scrubbed for publication).")

            # Audit: capture full report generation
            if self._audit:
                self._audit.record_tool_call(
                    tool="generate_report",
                    args={"title": session.goal},
                    result=result,
                    decision_source="report_phase",
                )

            # Auto-export PDF if configured
            if self._config.auto_pdf and session.report:
                try:
                    from emet.export.pdf import PDFReport
                    from emet.export.markdown import InvestigationReport

                    all_entities = []
                    for f in session.findings:
                        all_entities.extend(f.entities)

                    report_data = InvestigationReport(
                        title=session.goal,
                        summary=session.report,
                        entities=all_entities,
                        data_sources=[
                            {"name": t["tool"], "query": str(t.get("args", {}))}
                            for t in session.tool_history
                        ],
                    )

                    out_dir = Path(
                        self._config.output_dir
                        or self._config.memory_dir
                        or "investigations"
                    ) / "reports"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Sanitize filename from goal
                    safe_name = "".join(
                        c if c.isalnum() or c in " -_" else "_"
                        for c in session.goal[:60]
                    ).strip().replace(" ", "_")
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
                    pdf_path = out_dir / f"{safe_name}_{ts}.pdf"

                    pdf = PDFReport()
                    pdf.generate(report=report_data, output_path=str(pdf_path))

                    session._pdf_path = str(pdf_path)
                    session.record_reasoning(f"PDF report saved: {pdf_path}")
                except Exception as exc:
                    logger.debug("Auto-PDF export failed: %s", exc)

            # Record cost summary if LLM was used
            if self._cost_tracker:
                cost_info = self._cost_tracker.summary()
                session.record_reasoning(
                    f"LLM cost: ${cost_info['cumulative']:.4f} "
                    f"({cost_info['call_count']} calls, "
                    f"${cost_info['remaining']:.4f} budget remaining)"
                )

        except Exception as exc:
            session.record_reasoning(f"Report generation failed: {exc}")

    async def _llm_synthesize_report(self, session: Session) -> str | None:
        """Use the LLM to synthesize findings into a narrative report.

        Returns the markdown report text, or None if LLM unavailable.
        """
        client = self._get_llm_client()
        if client is None:
            return None

        # Don't use LLM synthesis for stub provider — it produces canned text
        try:
            from emet.cognition.llm_base import LLMProvider
            if hasattr(client, 'provider') and client.provider == LLMProvider.STUB:
                return None
        except Exception:
            pass

        # Build synthesis prompt from session data
        findings_text = "\n".join(
            f"- [{f.source}] (confidence: {f.confidence:.0%}) {f.summary}"
            for f in session.findings
        )

        entities_text = "\n".join(
            f"- [{entity.get('schema', '?')}] "
            f"{entity.get('properties', {}).get('name', [eid])[0] if entity.get('properties', {}).get('name') else eid}"
            for eid, entity in list(session.entities.items())[:30]
        )

        open_leads = session.get_open_leads()
        leads_text = "\n".join(
            f"- [{l.priority:.0%}] {l.description}"
            for l in open_leads[:10]
        ) if open_leads else "None — all leads resolved."

        prompt = f"""Synthesize the following investigation findings into a clear, structured report.

INVESTIGATION GOAL: {session.goal}

FINDINGS ({session.finding_count}):
{findings_text or "No findings."}

KEY ENTITIES ({session.entity_count}):
{entities_text or "No entities identified."}

OPEN LEADS:
{leads_text}

INVESTIGATION STATS:
- Turns used: {session.turn_count}
- Tools used: {', '.join(set(t['tool'] for t in session.tool_history))}

Write a markdown report with these sections:
## Summary
A 2-3 sentence executive summary of what was found.

## Key Findings
The most important discoveries, with confidence levels.

## Entity Network
Who/what was identified and how they connect.

## Open Questions
What remains unresolved — leads not yet pursued.

## Methodology
Brief note on tools and sources used.

Be factual. Only report what the findings support. Flag low-confidence items explicitly."""

        try:
            response = await client.complete(
                prompt,
                system="You are a report writer for investigative journalists. Write clear, factual, well-structured reports. Never fabricate details beyond what the findings state.",
                max_tokens=2048,
                temperature=0.3,
                tier="balanced",
            )
            report = response.text.strip()
            if len(report) > 100:  # Sanity check
                logger.info(
                    "LLM synthesized report: %d chars, %d input tokens, %d output tokens",
                    len(report), response.input_tokens, response.output_tokens,
                )
                return report
        except Exception as exc:
            logger.debug("LLM report synthesis failed: %s", exc)

        return None

    def _generate_investigation_graph(self, session: Session) -> None:
        """Build a relationship graph from investigation findings.

        Uses emet.graph to create a NetworkX graph of all entities and
        their relationships discovered during the investigation.
        """
        try:
            from emet.graph.engine import GraphEngine

            # Collect all entities across findings
            all_entities = list(session.entities.values())
            if not all_entities:
                session.record_reasoning("Graph: no entities to graph")
                return

            engine = GraphEngine()
            graph_result = engine.build_from_entities(all_entities)

            session._investigation_graph = graph_result
            session.record_reasoning(
                f"Graph: {graph_result.stats.nodes_loaded} nodes, "
                f"{graph_result.stats.edges_loaded} edges"
            )

        except Exception as exc:
            session.record_reasoning(f"Graph generation failed: {exc}")

    def _recall_prior_intelligence(
        self, session: Session
    ) -> list[dict[str, Any]]:
        """Recall findings from past investigations with overlapping entities.

        Scans saved sessions in memory_dir for entity name overlap with
        the current investigation goal. Returns relevant prior findings
        that the LLM can use for context.
        """
        from emet.agent.persistence import list_sessions, load_session

        memory_dir = Path(self._config.memory_dir)
        if not memory_dir.exists():
            return []

        goal_words = set(session.goal.lower().split())
        prior_findings: list[dict[str, Any]] = []

        try:
            past_sessions = list_sessions(memory_dir)
        except Exception:
            return []

        for meta in past_sessions[:50]:  # Cap to avoid slow scans
            # Quick relevance check: goal word overlap
            past_goal = meta.get("goal", "").lower()
            overlap = goal_words & set(past_goal.split())
            if len(overlap) < 1:
                continue

            # Don't recall from this exact session
            if meta.get("session_id") == session.id:
                continue

            try:
                past = load_session(meta["path"])
                for finding in past.findings:
                    # Only include substantive findings
                    if finding.confidence >= 0.5 and finding.summary:
                        prior_findings.append({
                            "source_session": meta.get("session_id", ""),
                            "source_goal": meta.get("goal", ""),
                            "source": finding.source,
                            "summary": finding.summary,
                            "confidence": finding.confidence,
                            "entity_count": len(finding.entities),
                        })
            except Exception:
                continue

        # Deduplicate by summary
        seen = set()
        unique: list[dict[str, Any]] = []
        for f in prior_findings:
            key = f["summary"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique[:10]  # Cap at 10 most relevant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_finding_summary(
    tool: str, action: dict[str, Any], result: dict[str, Any]
) -> str:
    """Build a human-readable summary of a tool result."""
    args = action.get("args", {})

    if tool == "search_entities":
        count = result.get("result_count", 0)
        return f"Entity search found {count} results for '{args.get('query', '?')}'"
    elif tool == "screen_sanctions":
        matches = result.get("match_count", len(result.get("matches", [])))
        count = result.get("screened_count", 1)
        return f"Sanctions screening: {matches} matches across {count} entities"
    elif tool == "trace_ownership":
        found = result.get("entities_found", len(result.get("entities", [])))
        depth = result.get("max_depth", 0)
        return f"Ownership trace found {found} entities (depth {depth}) for '{args.get('entity_name', '?')}'"
    elif tool == "osint_recon":
        return f"OSINT recon on '{args.get('target', '?')}'"
    elif tool == "monitor_entity":
        articles = result.get("article_count", 0)
        return f"Found {articles} news articles about '{args.get('entity_name', '?')}'"
    elif tool == "investigate_blockchain":
        chain = result.get("chain", "unknown")
        addr = args.get("address", "?")[:20]
        return f"Blockchain ({chain}): investigated {addr}"
    elif tool == "analyze_graph":
        algo = result.get("algorithm", args.get("algorithm", "network"))
        return f"Graph analysis: {algo}"
    elif tool == "generate_report":
        return f"Report generated: {result.get('title', args.get('title', 'investigation'))}"
    else:
        return f"{tool}: completed"


def _estimate_confidence(result: dict[str, Any]) -> float:
    """Rough confidence estimate from result quality."""
    if result.get("matches"):
        return 0.85
    if result.get("result_count", 0) > 0:
        return 0.7
    if result.get("entities"):
        return 0.6
    if result.get("data"):
        return 0.6
    return 0.4
