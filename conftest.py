"""Project-wide pytest configuration.

On Windows, asyncio defaults to WindowsProactorEventLoopPolicy which deadlocks
when anyio's TestClient portal tries to spin up a second event loop from a
background thread.  Switching to WindowsSelectorEventLoopPolicy fixes this.
"""
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

