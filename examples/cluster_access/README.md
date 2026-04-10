# Fake Slurm Cluster — OpenShell + OpenClaw Use Case

A toy example for **OpenShell → OpenClaw** deployments. Demonstrates cluster access
and job-launch workflows through a self-contained fake HPC cluster — no real machine
required.

The MCP server runs on the host. The OpenClaw agent runs inside the sandbox and talks
to it over HTTP.

```
┌─────────────────────────────────────┐      HTTP :9000      ┌──────────────────────────────┐
│           HOST MACHINE              │ ◄──────────────────► │     OPENSHELL SANDBOX        │
│                                     │                       │                              │
│  fake_cluster_mcp_server.py         │                       │  access_cluster_mcp_client.py│
│  binds 0.0.0.0:9000                 │                       │  connects to host IP:9000    │
│  LLM (ChatNVIDIA) lives here        │                       │  OpenClaw agent lives here   │
└─────────────────────────────────────┘                       └──────────────────────────────┘
```

---

## Repository layout

```
cluster_access/
├── fake_cluster_mcp_server.py        # MCP server — Slurm tools + LLM dispatcher  [HOST]
├── access_cluster_mcp_client.py      # MCP client — NL REPL for the sandbox        [SANDBOX]
├── sandbox_policy.yaml               # Sandbox egress policy (MCP server + PyPI + GitHub)
├── fake_cluster_server.py            # Paramiko SSH server (legacy, no LLM)
├── access_cluster_fake.py            # Paramiko client (legacy, no LLM)
└── slurm-cluster-mcp/               # Agent skill for interacting with the cluster
```

---

## Prerequisites

- [OpenShell](https://github.com/NVIDIA/OpenShell) installed
- Docker daemon running
- `uv` installed — the Python package manager used in this example

Install `uv` if needed:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options.

---

## Step 1 — Set up OpenShell + OpenClaw

### 1.1 Export your API keys

The MCP server on the host requires an NVIDIA API key for its LLM dispatcher:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

For OpenClaw inside the sandbox, export the key for your chosen inference provider.
See [OpenShell's provider list](https://github.com/NVIDIA/OpenShell?tab=readme-ov-file#providers) for supported options and their required environment variables.

### 1.2 Start the gateway

```bash
openshell gateway start
```

If no gateway exists, this bootstraps the full cluster locally (requires Docker).

### 1.3 Create the inference provider

Follow the [OpenShell provider setup guide](https://docs.nvidia.com/openshell/latest/sandboxes/manage-providers) to register the provider whose key you exported in step 1.1.

### 1.4 Verify the inference config

```bash
openshell inference get
```

### 1.5 Create the sandbox with OpenClaw

```bash
openshell sandbox create \
  --from openclaw \
  --forward 18789
```

This command connects you directly into the sandbox. Keep this terminal open.

To reconnect to an existing sandbox later:

```bash
openshell sandbox connect <sandbox-name>
```

### 1.6 Set the sandbox policy (open a new terminal)

This example ships with `sandbox_policy.yaml`, which grants the sandbox egress to:

- The MCP server at `host.openshell.internal` on ports 9000–9004
- Anthropic API (`api.anthropic.com`) for the OpenClaw agent
- PyPI and GitHub for package installs inside the sandbox
- NVIDIA inference endpoint (`integrate.api.nvidia.com`)

Review the file and adjust it for your environment, then apply:

```bash
openshell policy set <sandbox-name> \
  --policy examples/cluster_access/sandbox_policy.yaml \
  --wait
```

`--wait` blocks until the sandbox confirms the new policy is active. No restart required.

### 1.7 Onboard OpenClaw inside the sandbox

From the sandbox terminal (step 1.5), follow the [OpenClaw onboarding wizard](https://docs.openclaw.ai/start/wizard) to complete setup.

Once onboarded, you should see the chat UI URL. Click it or copy and paste the URL into your local PC's browser.

**Remote deployment:** If your OpenShell gateway runs on a remote machine, port 18789
is only reachable on that machine. Open an SSH tunnel from your local machine first:

```bash
ssh -L 18789:localhost:18789 <user>@<remote-host>
```

Then open the URL in your local browser.

---

## Step 2 — Start the MCP server on the HOST machine

### Prerequisites

Create a virtual environment and install dependencies
(see [uv environments](https://docs.astral.sh/uv/pip/environments/)):

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install fastmcp colorama python-dotenv langchain-core langchain-nvidia-ai-endpoints
```

`NVIDIA_API_KEY` must be set (see step 1.1). You can also write it to a `.env` file
(loaded automatically at startup):

```bash
echo 'NVIDIA_API_KEY=nvapi-...' > .env
```

### Find your host IP (the sandbox will need this)

```bash
# The IP reachable from Docker/container networks:
hostname -I | awk '{print $1}'
# or
ip route get 1 | awk '{print $7; exit}'
```

### Start the server


### Keep it alive across SSH disconnects (optional)

```bash
## ensure you are inside the venv where all the necessary python packages were installed 
tmux new-session -d -s fake-cluster \
  "cd examples/cluster_access && python fake_cluster_mcp_server.py"

## if you already created a tmux session previously, reattach to the existing session.
tmux attach -t fake-cluster   # watch logs
# Ctrl-B D to detach — server keeps running
```

---

## Step 3 — Run the client inside the sandbox

### Copy the client script into the sandbox

```bash
# From the host, copy the client into the sandbox (adjust sandbox name as needed):
openshell sandbox upload <sandbox-name> slurm-cluster-mcp /sandbox/.openclaw/workspace/skills/
```
Note: if openclaw is deployed successfully inside the sandbox in Step 1, this path /sandbox/.openclaw/workspace/skills/ should exist inside the sandbox

### Install dependencies inside the sandbox

```bash
uv venv
source .venv/bin/activate
uv pip install fastmcp colorama python-dotenv
```

### Point the client at the host server

The server URL must use the host's IP address as seen from the sandbox.

| Scenario | Address to use |
|---|---|
| OpenShell sandbox on same host | `http://host.openshell.internal:9000/mcp` (default) |
| Host has a LAN/public IP | `http://<host-ip>:9000/mcp` |
| Custom port | `http://<host-ip>:<port>/mcp` |

```bash
# Option A — environment variable (put in ~/.bashrc or sandbox .env)
export MCP_SERVER_URL="http://172.17.0.1:9000/mcp"
python access_cluster_mcp_client.py

# Option B — CLI flag
python access_cluster_mcp_client.py --server-url http://172.17.0.1:9000/mcp
```

### Example session

```
============================================================
  Fake Slurm Cluster — natural language interface
  Server : http://172.17.0.1:9000/mcp
  Type 'quit' or Ctrl-C to exit.
============================================================

cluster> what GPU partitions are available?
cluster> launch a training job with 4 GPUs for 10 epochs using vit-large
cluster> submit my train_bert.sh as a batch job
cluster> show me what jobs are running
cluster> what are my account limits?
cluster> how much compute have I used this month?
cluster> quit
```

---

## Available MCP tools

| Tool | What it does |
|---|---|
| `cluster_agent` | NL dispatcher — LLM maps any plain-English query to the right tool |
| `get_hostname` | Returns `dlcluster-headnode` |
| `sinfo` | Lists A100 / H100 / GB200 / CPU partitions (all idle) |
| `srun` | Simulates an interactive training job; streams epoch logs |
| `sbatch` | Submits a fake batch job; returns job ID |
| `squeue` | Shows the in-memory job table |
| `sacctmgr` | Returns user account association table |
| `sreport` | Returns fake CPU-minute utilisation report |

The LLM (`meta/llama-3.3-70b-instruct` via ChatNVIDIA) runs **server-side** —
the sandbox client sends only text queries and receives text results.
No API key is needed inside the sandbox.

---

## Troubleshooting

**`Connection refused` from sandbox:**
- Confirm the server is running: `curl http://<host-ip>:9000/mcp` from the host
- Check the host firewall allows port 9000: `sudo ufw allow 9000`
- Try the Docker bridge IP: `ip addr show docker0 | grep inet`

**`401 Unauthorized` on the server:**
- `NVIDIA_API_KEY` is missing or invalid — check the server's tmux session logs

**Wrong `host.openshell.internal` resolution:**
- Set `MCP_SERVER_URL` explicitly to the host's LAN IP instead of relying on the DNS alias

---

## Legacy: paramiko SSH server (no LLM, single machine)

```bash
# Terminal 1 — host
python fake_cluster_server.py

# Terminal 2 — host (or any machine that can SSH to localhost:2222)
python access_cluster_fake.py
```

No API key required.

### Host key

The server needs an RSA host key (`fake_host_key`). It is generated automatically on
first run — you don't need to create it manually. If you prefer to pre-generate it:

```bash
ssh-keygen -t rsa -b 2048 -f fake_host_key -N ""
```

`fake_host_key` is listed in `.gitignore` and should never be committed. Anyone who
clones the repo will have their own key generated on first run.
