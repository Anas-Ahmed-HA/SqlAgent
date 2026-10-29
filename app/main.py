"""Entry point: `python main.py` runs the service; `uvicorn main:app` serves it."""

from agent.host.app import app, run_server

__all__ = ["app"]


if __name__ == "__main__":
    run_server(app)
