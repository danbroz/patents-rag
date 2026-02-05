#!/usr/bin/env python3
"""
Deploy entrypoint (step 3 in the standard pipeline):

  1) python3 download.py
  2) python3 build.py
  3) python3 deploy.py

Python-only deploy script for the MCP server container.
"""

import os
import subprocess
import sys


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    image_name = "patents-rag-mcp:latest"
    container_name = "patents-rag-mcp"
    port = os.environ.get("PORT", "9000")

    def run(cmd: list[str]) -> int:
        print("+", " ".join(cmd))
        return subprocess.call(cmd, cwd=repo_root)

    # Build image
    rc = run(["docker", "build", "-t", image_name, repo_root])
    if rc != 0:
        return rc

    # Stop old container (ignore errors)
    _ = run(["docker", "rm", "-f", container_name])

    # Run new container
    rc = run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:9000",
            "-v",
            f"{repo_root}/patents-embeddings:/app/patents-embeddings:ro",
            "-v",
            f"{repo_root}/patents-index:/app/patents-index:ro",
            "-e",
            "PORT=9000",
            "-e",
            "FAISS_INDEX_PATH=/app/patents-embeddings/index.faiss",
            "-e",
            "TITLES_PATH=/app/patents-index/patent_titles.npy",
            image_name,
        ]
    )
    if rc != 0:
        return rc

    print("MCP server container is running.")
    print()
    print("Logs:")
    print(f"  docker logs -f {container_name}")
    print()
    print("SSE endpoint:")
    print(f"  http://localhost:{port}/sse")
    print()
    print("POST messages endpoint is announced via SSE 'endpoint' event.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

