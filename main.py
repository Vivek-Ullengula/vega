# main.py
"""Entry point for the Coaction Agent Platform."""

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else

import uvicorn
from coaction_agent_platform.app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "coaction_agent_platform.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
