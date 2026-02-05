from typing import Any

from mcp import types
from mcp.server.lowlevel import Server

from .search import search


def create_server() -> Server:
    app = Server("patents-rag-mcp")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="patent_search",
                title="Patent semantic search",
                description="Semantic search over the FAISS patent index (query -> top-k titles and scores).",
                inputSchema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "k": {"type": "integer", "description": "Number of results to return", "default": 10},
                    },
                },
            )
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name != "patent_search":
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                structuredContent={"error": "unknown_tool", "tool": name},
            )

        query = str(arguments.get("query", "")).strip()
        k_raw = arguments.get("k", 10)
        try:
            k = int(k_raw)
        except Exception:
            k = 10

        try:
            results = search(query=query, k=k)
        except Exception as e:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text=f"Search failed: {e}")],
                structuredContent={"error": "search_failed", "message": str(e)},
            )

        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Top {len(results)} results for query: {query}",
                )
            ],
            structuredContent={"query": query, "k": k, "results": results},
        )

    return app

