import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from mcp.server.sse import SseServerTransport

from .server import create_server


def create_starlette_app() -> Starlette:
    app = create_server()
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:  # type: ignore[reportPrivateUsage]
            await app.run(streams[0], streams[1], app.create_initialization_options())
        return Response()

    return Starlette(
        debug=bool(os.environ.get("DEBUG")),
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


app = create_starlette_app()

