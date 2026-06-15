# Orcasound AI MCP

This server lets you use AI (like Claude) to explore Orcasound data. It helps you find whale calls, check hydrophones, and prepare data for training without writing code.

## Requirements

To run this server, you need:
*   **Python 3.11+**: The core language (matches CI).
*   **Node.js**: Needed to run the "Inspector" or to connect to some AI clients.
*   **uv** (Recommended): For fast and clean setup.
*   **An AI Client**: Like Claude Desktop, Gemini, or VS Code (with an MCP plugin).

> [!NOTE]
> The base `requirements-mcp.txt` installs a minimal set of dependencies for basic data interrogation. Running the model comparison tool (`compare_models_on_clip`) requires the full ML/audio dependencies from the main `requirements.txt`.

## How it works

When you connect this server to an AI (like Gemini or Claude), the AI "reads" the names and descriptions of our tools. 

Because the tool names (like `get_recent_detections`) and their instructions are **fixed** in the code, the AI doesn't guess. It knows exactly what data it's getting and how to use it. This makes the conversation reliable and consistent every time.

---

## Quick Start 

We use `uv` because it's the fastest way to set everything up.

1. **Install uv**:
   For Unix:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```

   For Windows:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Set up the tools**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements-mcp.txt
   ```

3. **Test it (The "Inspector")**:
   Run this to see a web page with all the tools:
   ```bash
   export PYTHONPATH=src
   export DANGEROUSLY_OMIT_AUTH=true
   npx -y @modelcontextprotocol/inspector .venv/bin/python src/mcp_server.py
   ```
   Open [http://localhost:6274](http://localhost:6274) and click **Connect**.

---

## Real Example: Finding new data

Here is exactly how this helps research. I asked an AI ( Gemini-cli): 
> *"Is there any new whale data at Sunset Bay we haven't trained on yet?"*

**The AI did the following automatically:**
1. Looked up the Sunset Bay station.
2. Checked the latest detections from the live website.
3. Compared them to our local training files.
4. Found **31 new whale calls** that were missing!
5. Exported them to `unlabeled_sunset_bay.csv` so I could start using them.

**Example of the generated data:**
```csv
det_032veqonCsn1LcaZ81ifff,2026-04-07T20:09:18.000000Z,human,vessel,Vessel hitting the hydrophone,47340.364191,1775545218
det_032veAu4cz2IRFOO48S6LU,2026-04-07T19:41:44.000000Z,human,other,a train,45685.561536,1775545218
det_032ve9oSUoGKDvj1ClujUY,2026-04-07T19:41:15.000000Z,human,other,A Train,45656.904129,1775545218
det_032uuCAj8ydlSPw76yrYjQ,2026-04-06T12:27:47.000000Z,machine,whale,,19650.0,1775458817
det_032rv284XNvcWjwk6O8VKc,2026-04-01T06:42:30.000000Z,machine,whale,Transient calls,85332.0,1774940418
det_032rcOgki2tf0KtWxR3Vu4,2026-03-31T18:04:00.000000Z,human,whale,,39821.773909,1774940418
```

**Total time:** 15 seconds. (Doing this manually would take more and we would need to run more than one different scripts and SQL queries).

---

## Using it with Claude Desktop

To use these tools inside Claude, add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orcasound": {
      "command": "/absolute/path/to/pods-ai/.venv/bin/python",
      "args": ["/absolute/path/to/pods-ai/src/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/pods-ai/src"
      }
    }
  }
}
```

Now you can just ask Claude: *"Find me new whale calls at Orcasound Lab and save them to a CSV."*
