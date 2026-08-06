"""Emet MCP Server -- investigative intelligence via Model Context Protocol.

Exposes Project Emet's OCCRP-style investigative capabilities as MCP
tools for Claude Code, Claude Desktop, or any MCP-compatible client.

Tools:
  emet_investigate     -- Run an investigation query (agent loop)
  emet_search_entities -- Federated entity search (people, companies, addresses)
  emet_trace_network   -- Trace connections between entities (ownership, graph)
  emet_foia            -- Generate a FOIA request for a topic/agency
  emet_health          -- Agent and system status

Usage:
  python emet_mcp_server.py --config emet.json

  # In Claude Code .claude/mcp.json:
  {
    "mcpServers": {
      "emet": {
        "command": "python",
        "args": ["emet_mcp_server.py"]
      }
    }
  }

Copyright Liberation Labs / TH Coalition.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("emet.mcp_server")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
    )
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

HAS_EMET = False
try:
    from emet.mcp.tools import EmetToolExecutor
    from emet.agent.health import HealthCheck
    from emet.graph.engine import GraphEngine
    HAS_EMET = True
except ImportError:
    pass


def _json_text(data: Any) -> list["TextContent"]:
    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# FOIA request generator
# ---------------------------------------------------------------------------

_FOIA_TEMPLATE = """\
Via: Freedom of Information Act, 5 U.S.C. {statute}

{date}

{agency}
FOIA/PA Mail Referral Unit
{address}

Re: Freedom of Information Act Request -- {topic}

Dear FOIA Officer,

Pursuant to the Freedom of Information Act (FOIA), {statute_full}, \
I respectfully request access to the following records:

{description}

Date range: {date_range}

I request that responsive records be provided in electronic format \
where available. If any responsive records are classified or otherwise \
exempt, I request that you segregate and release all reasonably \
segregable, non-exempt portions.

I am willing to pay reasonable search and duplication fees up to \
${fee_limit:.2f}. Please notify me before incurring costs above \
that amount.

If you deny any part of this request, please cite the specific \
exemption(s) and notify me of the appeal procedures available.

Thank you for your prompt attention to this request.

Sincerely,
[YOUR NAME]
[YOUR ORGANIZATION]
[YOUR ADDRESS]
[YOUR EMAIL]
"""

_AGENCY_ADDRESSES: dict[str, dict[str, str]] = {
    "DOJ": {
        "name": "U.S. Department of Justice",
        "address": "950 Pennsylvania Avenue, NW\nWashington, DC 20530-0001",
    },
    "SEC": {
        "name": "U.S. Securities and Exchange Commission",
        "address": "100 F Street, NE\nWashington, DC 20549",
    },
    "FBI": {
        "name": "Federal Bureau of Investigation",
        "address": "Record/Information Dissemination Section\n170 Marcel Drive\nWinchester, VA 22602-4843",
    },
    "EPA": {
        "name": "U.S. Environmental Protection Agency",
        "address": "National FOIA Office\n1200 Pennsylvania Avenue, NW (2822T)\nWashington, DC 20460",
    },
    "DOD": {
        "name": "U.S. Department of Defense",
        "address": "Office of Freedom of Information\n1155 Defense Pentagon\nWashington, DC 20301-1155",
    },
    "STATE": {
        "name": "U.S. Department of State",
        "address": "Office of Information Programs and Services\nA/GIS/IPS/RL\nSA-2, Suite 8100\nWashington, DC 20522-0208",
    },
    "TREASURY": {
        "name": "U.S. Department of the Treasury",
        "address": "FOIA and Transparency\n1500 Pennsylvania Avenue, NW\nWashington, DC 20220",
    },
}


def _generate_foia(
    topic: str,
    agency: str = "",
    description: str = "",
    date_range: str = "",
    fee_limit: float = 25.0,
) -> dict[str, Any]:
    agency_key = agency.upper().strip()
    agency_info = _AGENCY_ADDRESSES.get(agency_key)

    if agency_info:
        agency_name = agency_info["name"]
        agency_addr = agency_info["address"]
    elif agency:
        agency_name = agency
        agency_addr = "[AGENCY ADDRESS]"
    else:
        agency_name = "[AGENCY NAME]"
        agency_addr = "[AGENCY ADDRESS]"

    if not description:
        description = (
            f"All records, communications, memoranda, emails, reports, "
            f"and other documents related to: {topic}"
        )

    if not date_range:
        date_range = "January 1, 2020 to present"

    letter = _FOIA_TEMPLATE.format(
        statute="\\u00a7 552",
        statute_full="5 U.S.C. \\u00a7 552",
        date=datetime.now(timezone.utc).strftime("%B %d, %Y"),
        agency=agency_name,
        address=agency_addr,
        topic=topic,
        description=description,
        date_range=date_range,
        fee_limit=fee_limit,
    )

    return {
        "topic": topic,
        "agency": agency_name,
        "agency_code": agency_key if agency_info else None,
        "known_agency": agency_info is not None,
        "fee_limit": fee_limit,
        "date_range": date_range,
        "letter": letter,
        "supported_agencies": list(_AGENCY_ADDRESSES.keys()),
        "generated_at": _ts(),
        "note": (
            "This is a template. Review and customize before sending. "
            "Add your identity, adjust the description, and verify the "
            "agency address is current."
        ),
    }


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(config_path: str = "") -> "Server":
    if not HAS_MCP:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install mcp"
        )

    config: dict[str, Any] = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)

    server = Server("emet")
    executor = EmetToolExecutor() if HAS_EMET else None

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="emet_investigate",
                description=(
                    "Run an OCCRP-style investigation query. Performs federated "
                    "entity search, graph analysis, sanctions screening, and "
                    "ownership tracing in a multi-step agent loop. Returns "
                    "structured findings with provenance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Investigation goal or question",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Limit to specific data sources: aleph, "
                                "opensanctions, opencorporates, icij, gleif. "
                                "Empty = all available."
                            ),
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "description": "Max depth for ownership/network tracing",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="emet_search_entities",
                description=(
                    "Federated entity search across Aleph, OpenSanctions, "
                    "OpenCorporates, ICIJ Offshore Leaks, and GLEIF. Returns "
                    "FtM entities with provenance. Supports Person, Company, "
                    "and Address queries."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Entity name or search term",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ["Person", "Company", "Address", "Any"],
                            "default": "Any",
                            "description": "Filter by entity type",
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Limit to specific sources: aleph, opensanctions, "
                                "opencorporates, icij, gleif. Empty = all."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Max results per source",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="emet_trace_network",
                description=(
                    "Trace connections between entities: corporate ownership "
                    "chains, beneficial ownership, directorships, and hidden "
                    "network links. Runs graph algorithms (community detection, "
                    "centrality, brokers) on the entity network."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Starting entity name to trace from",
                        },
                        "trace_type": {
                            "type": "string",
                            "enum": ["ownership", "network", "full"],
                            "default": "full",
                            "description": (
                                "ownership = corporate chain only, "
                                "network = graph algorithms only, "
                                "full = both"
                            ),
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "description": "Max chain depth for ownership tracing",
                        },
                        "include_officers": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include directors and officers",
                        },
                    },
                    "required": ["entity_name"],
                },
            ),
            Tool(
                name="emet_foia",
                description=(
                    "Generate a FOIA (Freedom of Information Act) request "
                    "letter for a given topic and agency. Supports DOJ, SEC, "
                    "FBI, EPA, DOD, STATE, TREASURY. Returns a formatted "
                    "request template ready for review and submission."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Subject of the records request",
                        },
                        "agency": {
                            "type": "string",
                            "description": (
                                "Target agency code (DOJ, SEC, FBI, EPA, DOD, "
                                "STATE, TREASURY) or full agency name"
                            ),
                            "default": "",
                        },
                        "description": {
                            "type": "string",
                            "description": "Specific records description (optional, auto-generated if empty)",
                            "default": "",
                        },
                        "date_range": {
                            "type": "string",
                            "description": "Date range for records (e.g. 'January 2020 to present')",
                            "default": "",
                        },
                        "fee_limit": {
                            "type": "number",
                            "default": 25.0,
                            "description": "Maximum fee willing to pay without notification",
                        },
                    },
                    "required": ["topic"],
                },
            ),
            Tool(
                name="emet_health",
                description=(
                    "Agent health check: API key status, data freshness, "
                    "disk space, dependency availability, and investigation "
                    "session summary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "emet_investigate":
            return await _handle_investigate(arguments)
        elif name == "emet_search_entities":
            return await _handle_search_entities(arguments)
        elif name == "emet_trace_network":
            return await _handle_trace_network(arguments)
        elif name == "emet_foia":
            return await _handle_foia(arguments)
        elif name == "emet_health":
            return await _handle_health(arguments)
        else:
            return _json_text({"error": f"Unknown tool: {name}"})

    async def _handle_investigate(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        sources = args.get("sources") or []
        max_depth = args.get("max_depth", 3)

        if not HAS_EMET or executor is None:
            return _json_text({
                "status": "unavailable",
                "query": query,
                "error": (
                    "Emet core not installed. Install with: "
                    "pip install -e /path/to/Project-Emet"
                ),
            })

        results: dict[str, Any] = {"query": query, "steps": []}

        search_result = await executor.execute_raw(
            "search_entities",
            {"query": query, "sources": sources, "limit": 20},
        )
        results["search"] = {
            "result_count": search_result.get("result_count", 0),
            "source_stats": search_result.get("source_stats", {}),
            "errors": search_result.get("errors", []),
        }
        results["steps"].append("search_entities")

        entities = search_result.get("entities", [])

        if entities:
            try:
                ownership = await executor.execute_raw(
                    "trace_ownership",
                    {"entity_name": query, "max_depth": max_depth},
                )
                results["ownership"] = {
                    "entities_found": ownership.get("entities_found", 0),
                    "owners": ownership.get("owners", []),
                    "cycles_detected": ownership.get("cycles_detected", False),
                }
                results["steps"].append("trace_ownership")
            except Exception as exc:
                results["ownership"] = {"error": str(exc)}

        if entities:
            screen_input = [
                {"name": _extract_name(e), "schema": e.get("schema", "Person")}
                for e in entities[:10]
                if _extract_name(e)
            ]
            if screen_input:
                try:
                    sanctions = await executor.execute_raw(
                        "screen_sanctions",
                        {"entities": screen_input, "threshold": 0.7},
                    )
                    results["sanctions"] = {
                        "screened_count": sanctions.get("screened_count", 0),
                        "match_count": sanctions.get("match_count", 0),
                        "matches": sanctions.get("matches", []),
                    }
                    results["steps"].append("screen_sanctions")
                except Exception as exc:
                    results["sanctions"] = {"error": str(exc)}

        results["entity_count"] = len(entities)
        results["timestamp"] = _ts()
        return _json_text(results)

    async def _handle_search_entities(args: dict) -> list[TextContent]:
        query = args.get("query", "")
        entity_type = args.get("entity_type", "Any")
        sources = args.get("sources") or []
        limit = args.get("limit", 20)

        if not HAS_EMET or executor is None:
            return _json_text({
                "status": "unavailable",
                "query": query,
                "error": "Emet core not installed.",
            })

        result = await executor.execute_raw(
            "search_entities",
            {
                "query": query,
                "entity_type": entity_type,
                "sources": sources,
                "limit": limit,
            },
        )
        return _json_text(result)

    async def _handle_trace_network(args: dict) -> list[TextContent]:
        entity_name = args.get("entity_name", "")
        trace_type = args.get("trace_type", "full")
        max_depth = args.get("max_depth", 3)
        include_officers = args.get("include_officers", True)

        if not HAS_EMET or executor is None:
            return _json_text({
                "status": "unavailable",
                "entity_name": entity_name,
                "error": "Emet core not installed.",
            })

        result: dict[str, Any] = {
            "entity_name": entity_name,
            "trace_type": trace_type,
        }

        if trace_type in ("ownership", "full"):
            try:
                ownership = await executor.execute_raw(
                    "trace_ownership",
                    {
                        "entity_name": entity_name,
                        "max_depth": max_depth,
                        "include_officers": include_officers,
                    },
                )
                result["ownership"] = ownership
            except Exception as exc:
                result["ownership"] = {"error": str(exc)}

        if trace_type in ("network", "full"):
            try:
                search = await executor.execute_raw(
                    "search_entities",
                    {"query": entity_name, "limit": 30},
                )
                entities = search.get("entities", [])
                if entities:
                    graph = await executor.execute_raw(
                        "analyze_graph",
                        {"entities": entities, "algorithm": "full"},
                    )
                    result["network"] = graph
                else:
                    result["network"] = {
                        "node_count": 0,
                        "note": "No entities found for network analysis",
                    }
            except Exception as exc:
                result["network"] = {"error": str(exc)}

        result["timestamp"] = _ts()
        return _json_text(result)

    async def _handle_foia(args: dict) -> list[TextContent]:
        result = _generate_foia(
            topic=args.get("topic", ""),
            agency=args.get("agency", ""),
            description=args.get("description", ""),
            date_range=args.get("date_range", ""),
            fee_limit=args.get("fee_limit", 25.0),
        )
        return _json_text(result)

    async def _handle_health(args: dict) -> list[TextContent]:
        result: dict[str, Any] = {
            "server": "emet-mcp",
            "version": "0.10.0",
            "timestamp": _ts(),
            "emet_available": HAS_EMET,
            "mcp_sdk": True,
        }

        if HAS_EMET:
            try:
                report = HealthCheck.run()
                result["status"] = "healthy" if report.error_count == 0 else "degraded"
                result["ok"] = report.ok_count
                result["warnings"] = report.warning_count
                result["errors"] = report.error_count
                result["checks"] = [
                    {
                        "name": c.name,
                        "status": c.status,
                        "message": c.message,
                        "suggestion": c.suggestion,
                    }
                    for c in report.checks
                ]
            except Exception as exc:
                result["status"] = "error"
                result["health_error"] = str(exc)
        else:
            result["status"] = "core_unavailable"
            result["note"] = (
                "Emet core is not installed. Health checks require: "
                "pip install -e /path/to/Project-Emet"
            )

        return _json_text(result)

    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="emet://config",
                name="Emet Configuration",
                description="Current configuration (non-sensitive fields)",
                mimeType="application/json",
            ),
            Resource(
                uri="emet://tools",
                name="Available Tools",
                description="List of investigative tools and their capabilities",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str):
        if uri == "emet://config":
            cfg = dict(config) if config else {}
            cfg["emet_available"] = HAS_EMET
            cfg["server_version"] = "0.10.0"
            return json.dumps(cfg, indent=2)
        elif uri == "emet://tools":
            tools = [
                {"name": "emet_investigate", "category": "investigation"},
                {"name": "emet_search_entities", "category": "search"},
                {"name": "emet_trace_network", "category": "analysis"},
                {"name": "emet_foia", "category": "legal"},
                {"name": "emet_health", "category": "system"},
            ]
            return json.dumps({"tools": tools}, indent=2)
        else:
            return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


def _extract_name(entity: dict[str, Any]) -> str:
    props = entity.get("properties", {})
    names = props.get("name", [])
    if isinstance(names, list) and names:
        return names[0]
    if isinstance(names, str):
        return names
    return ""


async def main(config_path: str = ""):
    server = create_server(config_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    if not HAS_MCP:
        print(
            "Error: MCP SDK not installed.\n"
            "Install with: pip install mcp\n"
            "Then re-run: python emet_mcp_server.py",
            file=sys.stderr,
        )
        sys.exit(1)

    import asyncio

    parser = argparse.ArgumentParser(description="Emet MCP Server")
    parser.add_argument(
        "--config",
        default="",
        help="Path to JSON configuration file",
    )
    parsed = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(main(parsed.config))
