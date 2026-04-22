# MSI Access Guide for New Students
### Minnesota Supercomputing Institute (Agate Cluster)
*Prepared for onboarding to the Coughlin Lab / SLSN pipeline project*

---

## Before You Start: How MSI Is Organized

Before logging in or running anything, it helps to understand how the system is structured. MSI is not like your laptop — it is a shared computing cluster used by thousands of researchers simultaneously. Understanding this structure will make everything else in this guide make sense.

### The Mental Model: A Restaurant Kitchen

Think of MSI like a restaurant:

- **You** are a customer placing an order
- **The login node** is the front desk — where you arrive, look at the menu, place your order
- **SLURM** is the order ticket system — it takes your order and routes it to the right station
- **The compute nodes** are the kitchen — where the actual cooking (computation) happens

You don't walk into the kitchen and start cooking yourself. You place your order at the front, and the kitchen handles it. That's exactly how MSI works.

### Two Types of Nodes

Every time you connect to MSI, you land on one of two types of machines:

| Node Type | What it is | What it's for |
|-----------|-----------|---------------|
| **Login node** | A shared machine everyone is logged into simultaneously | Navigation, editing files, submitting jobs |
| **Compute node** | A dedicated machine allocated just to you | Running actual computations |

**The login node is for:**
- Navigating files (`cd`, `ls`, `cat`)
- Editing code (in VS Code, `nano`, `vim`)
- `git` commands (pull, push, commit)
- Submitting and monitoring SLURM jobs
- Downloading files with `wget`
- Light, fast commands that finish in seconds

**The login node is NOT for:**
- Running a Python script that takes more than a minute or two
- Running Jupyter notebooks with heavy computations
- Processing large datasets directly

Because it is shared, running heavy code on the login node slows it down for every researcher on the cluster at that moment. MSI will sometimes kill processes that abuse the login node.

### What SLURM Is

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduler. It is not the computation itself — it is the system that routes your computation to the right place. When you request a compute node (either through a browser form or a script), SLURM puts you in a queue, waits for resources to be available, and then runs your code on a compute node.

There are two ways to access a compute node:

| Method | Best for | How |
|--------|----------|-----|
| **Open OnDemand** (browser form) | Interactive work — Jupyter notebooks, data exploration, plotting | Fill out a form, click Launch |
| **`sbatch` script** (terminal) | Long unattended runs — full pipeline jobs overnight | Write a `.slurm` script, submit from terminal |

Both methods talk to SLURM. Both land you on a compute node. The difference is just the interface. This guide covers both.

---

## Step 0: Get Added to the MSI PI Group

**Michael must add you to the MSI PI group before you can log in.**

MSI organizes access by PI group — you cannot connect until your faculty advisor has added your UMN Internet ID to their group via the MyMSI portal. Remind Michael to do this before trying anything below.

Once added, you will receive a confirmation email. Your MSI username is your UMN Internet ID (x500), e.g., `smith123`.

---

## Step 1: On Campus or Off Campus?

Your location determines what you need before connecting.

| Location | Network | What you need |
|----------|---------|---------------|
| On campus | eduroam WiFi | Nothing extra — you're on the UMN network |
| Off campus | Home or other WiFi | **Must connect to UMN VPN first** (see Section 2) |

> ⚠️ MSI will silently refuse connections from outside the UMN network — no error message, it just won't connect. If MSI suddenly stops responding from home, check that your VPN is still on.

---

## Section 2: VPN Setup (Off-Campus Access)

You need the **Cisco Secure Client** (formerly called AnyConnect). This is a free UMN-provided app.

### Install

1. Go to: **[it.umn.edu/vpn](https://it.umn.edu/services-technologies/virtual-private-network-vpn)**
2. Download the Cisco Secure Client for your OS (Mac, Windows, or Linux)
3. Install it — it appears as "Cisco Secure Client" in your applications

### Connect

1. Open **Cisco Secure Client**
2. Enter: **`vpn.umn.edu`**
3. When prompted for tunnel type, choose: **"UMN - Split Tunnel - General Access"**
4. Enter your UMN Internet ID and password
5. Approve the **Duo push** on your phone
6. Status bar says **"Connected"**

> 🔔 Sessions auto-disconnect after 60 minutes of idle and after 7 days total. If your connection to MSI drops unexpectedly, check VPN first.

### Disconnect When Done
VPN icon in menu bar/taskbar → **Disconnect**.

---

## Section 3: Two Ways to Connect to MSI

Once you're on the UMN network (on campus or via VPN):

| Method | Best for | What you need |
|--------|----------|---------------|
| **Open OnDemand** | Beginners, Jupyter notebooks, interactive work, no terminal setup | Just a browser |
| **SSH terminal** | Submitting SLURM jobs, scripting, VS Code workflow | Terminal or VS Code |

Both connect to the same cluster and the same files. They are different doors into the same house. Most researchers end up using both — OnDemand for interactive exploration, SSH for submitting production jobs.

---

## Section 4: SSH Access and Setup

SSH (Secure Shell) lets you connect to MSI from a terminal on your laptop. This is the method used for submitting SLURM batch jobs and is what most researchers use day-to-day.

### 4a: Connect for the First Time

**Mac / Linux** — open Terminal (search "Terminal" in Spotlight on Mac):
```bash
ssh your_x500@agate.msi.umn.edu
```

**Windows** — open PowerShell or Command Prompt:
```
ssh your_x500@agate.msi.umn.edu
```
Or download **PuTTY** ([putty.org](https://www.putty.org)), enter `agate.msi.umn.edu` as the Host Name, connection type SSH, click Open.

You will be prompted for your password, then a Duo push. You land on a **login node**.

---

### 4b: Set Up SSH Keys (Do This — Saves a Password Every Time)

SSH keys let you connect without typing your password. Before generating one, check if you already have one — you don't want to overwrite a key that other services (like GitHub) are already using.

**Step 1 — Check if a key already exists (run on your laptop):**
```bash
ls ~/.ssh/
```
If you see `id_ed25519` and `id_ed25519.pub` (or `id_rsa` / `id_rsa.pub`) — you already have a key. **Skip Step 2.**

If the folder is empty or doesn't exist, continue to Step 2.

**Step 2 — Generate a new key (only if you don't have one):**
```bash
ssh-keygen -t ed25519 -C "your_x500@umn.edu"
# Press Enter to accept the default file location
# Press Enter twice to skip the passphrase
```

**Step 3 — Copy your public key to MSI:**
```bash
ssh-copy-id your_x500@agate.msi.umn.edu
# Enter your MSI password one last time
```

**Step 4 — Test it:**
```bash
ssh your_x500@agate.msi.umn.edu
# Should connect without asking for a password
```

> ⚠️ Never share or move the `id_ed25519` file (no `.pub`). That is your private key. The `.pub` file is the one that goes on MSI — `ssh-copy-id` handles this automatically.

---

### 4c: Create an SSH Shortcut (Type `ssh msi` Instead of the Full Address)

Once keys are set up, you can create a shortcut so you only ever type `ssh msi`.

Open (or create) `~/.ssh/config` on your **laptop**:
```bash
nano ~/.ssh/config
```

Add this block:
```
Host msi
    HostName agate.msi.umn.edu
    User your_x500
    IdentityFile ~/.ssh/id_ed25519
```

Save and close. Now:
```bash
ssh msi
```

And `scp` works the same way:
```bash
scp ~/Downloads/myfile.csv msi:~/Documents/Research/Transient_Metrics/SLSNe_Metric/
```

> 💡 This also makes VS Code's SSH extension smoother — `msi` will appear as a saved remote in the connection menu.

---

### 4d: VS Code + SSH

If you connect to MSI through VS Code's SSH extension and submit SLURM jobs from the integrated terminal — **you are not causing a slowdown.** The terminal lands you on the login node, but `sbatch` is a lightweight command. The computation itself runs on a compute node — SLURM moves it there for you.

The only thing to avoid: running a heavy Python script directly in the VS Code terminal without going through SLURM (e.g., accidentally running `python run_slsn_pipeline.py` directly and letting it sit for 30 minutes).

---

## Section 5: Open OnDemand (Browser Access)

Open OnDemand is a web portal for interactive access to MSI. No terminal setup needed.

### Log In
Go to: **[ondemand.msi.umn.edu](https://ondemand.msi.umn.edu)** → UMN Internet ID + password + Duo.

### What You'll See
The dashboard menu bar has:
- **Files** — browse, upload, and download files on MSI
- **Jobs** — view running and queued SLURM jobs
- **Interactive Apps** — launch Jupyter, Desktop sessions, etc.
- **Clusters → Agate Shell Access** — a terminal in your browser (lands on the login node)

### Starting a Jupyter or Desktop Session

When you launch Jupyter or Desktop through OnDemand, you are filling out a SLURM resource request form. OnDemand submits it to SLURM on your behalf — you get a compute node without writing a script.

**To launch:**
1. Click **Interactive Apps → Jupyter** (or Desktop)
2. Fill in the resource form (see Section 6 for what each field means)
3. Click **Launch**
4. Wait for the session card to show **"Running"** (usually < 2 min)
5. Click **Connect to Jupyter** (or **Launch Desktop**)

> ✅ You are now running on a compute node, not a login node. Heavy notebook cells are fine here.

**Important:** If your browser tab closes or your laptop sleeps, the session may disconnect — but the SLURM job keeps running until the time limit. Go back to **My Interactive Sessions** to reconnect or delete the session when done. Always click **Delete** to release resources when finished, not just close the tab.

**When to use OnDemand vs `sbatch`:**

| You want to... | Use |
|----------------|-----|
| Explore data, iterate on plots, run notebook cells interactively | OnDemand Jupyter |
| Run the full pipeline unattended for 13+ hours | `sbatch` script |
| Quick terminal access without VS Code | OnDemand Shell |

---

## Section 6: Understanding SLURM Resources — What to Request

Both OnDemand and `.slurm` scripts ask for the same set of resources. This section explains what each parameter means and how to choose values sensibly, using our project as a concrete example.

### The Parameters

| Parameter | What it means | OnDemand field | `.slurm` directive |
|-----------|--------------|----------------|-------------------|
| **Account** | Which lab account to charge compute time to | Account dropdown | `#SBATCH --account=` |
| **Partition** | Which pool of compute nodes to draw from | Partitions dropdown | `#SBATCH --partition=` |
| **Nodes** | How many separate machines | Number of Nodes | `#SBATCH --nodes=` |
| **Cores** | CPU cores on each machine | Cores per Node | `#SBATCH --cpus-per-task=` |
| **Memory** | RAM per machine | Memory per Node | `#SBATCH --mem=` |
| **Time** | Wall clock limit — job is killed when this expires | Time Limit | `#SBATCH --time=` |
| **Scratch** | Temporary fast local disk | Scratch per Node | *(not commonly used)* |
| **GPUs** | GPU allocation | GPUs per Node | `#SBATCH --gres=gpu:` |

**For this project, always use:**
- Account: `cough052`
- Partition: `agsmall`
- Nodes: `1` (always — we are not doing MPI-parallel work across machines)

### How to Decide How Much to Request

The right values depend on what your job actually does. Here is how to think about it, using our pipeline as an example:

**Memory:**
The dominant cost is loading the simulated population into RAM. Our three rate models have different population sizes:

| Model | Population size | Approximate RAM per worker |
|-------|----------------|---------------------------|
| `fe_dependent` | ~2.2 million events | ~600 MB |
| `o_dependent` | ~1.1 million events | ~300 MB |
| `naive` | ~750k events | ~200 MB |

When running 4 cadences in parallel (4 workers), multiply by 4 and add overhead for MAF:
- `fe_dependent`: 4 × 600 MB + MAF overhead → **128G requested**
- `o_dependent` / `naive`: could run on less, but 128G is safe for all three

For an interactive Jupyter session doing exploratory analysis (not running the full pipeline), **8–32G is usually enough**.

**Cores:**
We run 4 cadences simultaneously, one per worker. That requires 4 cores. Requesting more cores than you have parallel workers wastes resources without speeding things up.

**Time:**
A full model × 4 cadences run takes approximately 13 hours. We request `24:00:00` as a buffer. Unused time is released when the job finishes — you are not penalized for requesting more than you use. But if you request too little and the job hits the wall time, SLURM kills it mid-run with no output saved.

> 💡 **Start with more time than you think you need.** You can always cancel early. You cannot recover a job killed by the time limit.

> 💡 **"Shorter times will probably start faster"** — the OnDemand form says this, and it's true. SLURM can slot a 1-hour job into gaps between longer jobs. If you only need 2 hours, don't request 24.

**Quick reference for this project:**

| Task | Cores | Memory | Time |
|------|-------|--------|------|
| Interactive notebook (exploration) | 2–4 | 8–32G | 2–4 hrs |
| Full pipeline, one model × 4 cadences | 4 | 128G | 24 hrs |
| Debugging a single cadence | 1 | 32G | 2 hrs |

---

## Section 7: Getting a Compute Node

There are three ways to get onto a compute node, depending on what you need.

### Option A: OnDemand (Interactive — Browser)

Fill out the form at [ondemand.msi.umn.edu](https://ondemand.msi.umn.edu) → Interactive Apps → Jupyter or Desktop. Described in Section 5. Best for notebook work.

### Option B: `srun` (Interactive — Terminal)

If you are already on the login node via SSH and want a quick interactive compute session without writing a script:

```bash
srun -N 1 --cpus-per-task=4 --mem=32G --time=2:00:00 -p agsmall --account=cough052 --pty bash
```

Your prompt changes to a compute node hostname (e.g., `acn001`). You are now on a compute node. Run scripts, test code, run Python directly — it is all fine here.

Use this for: quick tests, debugging a crash, running something that takes 30–60 minutes while you watch it.

> ⚠️ If your SSH connection drops or your laptop sleeps, the `srun` session dies and takes your computation with it. For anything over ~1 hour, use `sbatch` instead.

### Option C: `sbatch` (Batch — Runs Unattended)

Write a `.slurm` script and submit it. The job runs in the background on a compute node regardless of your connection. Described in full in Section 8.

Use this for: full pipeline runs, anything that takes multiple hours, anything you want to run overnight.

---

## Section 8: Submitting SLURM Batch Jobs

### What a `.slurm` Script Is

A `.slurm` script is a regular shell script with SLURM resource directives at the top. The directives start with `#SBATCH` — SLURM reads them, bash treats them as comments. Everything below the directives is the actual code your job runs.

### Minimal Example

```bash
#!/bin/bash
# -----------------------------------------------------------
# SLURM directives — SLURM reads these, bash ignores them
# -----------------------------------------------------------
#SBATCH --job-name=my_test          # Label shown in squeue
#SBATCH --account=cough052          # Lab account (always required)
#SBATCH --partition=agsmall         # Which queue to use
#SBATCH --nodes=1                   # Number of machines (always 1 for us)
#SBATCH --ntasks=1                  # Number of parallel tasks
#SBATCH --cpus-per-task=4           # CPU cores
#SBATCH --mem=32G                   # RAM
#SBATCH --time=08:00:00             # Wall time: HH:MM:SS
#SBATCH --output=logs/job_%j.out    # Stdout log (%j = job ID)
#SBATCH --error=logs/job_%j.err     # Stderr log
#SBATCH --mail-type=END,FAIL        # Email when done or failed
#SBATCH --mail-user=your_x500@umn.edu

# -----------------------------------------------------------
# Your actual job — regular bash from here down
# -----------------------------------------------------------
source /common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1

cd /users/1/your_x500/Documents/Research/YourProject
python run_my_script.py --some-flag
```

Always create your log directory before submitting:
```bash
mkdir -p logs
sbatch my_job.sh
```

SLURM responds immediately:
```
Submitted batch job 12345678
```

Your terminal is free. The job runs on a compute node in the background. You can close VS Code, close your laptop, go home — it keeps running.

### Required Fields — Never Skip These

```bash
#SBATCH --account=cough052     # Without this, job may fail or charge the wrong group
#SBATCH --partition=agsmall    # Without this, SLURM doesn't know which queue to use
#SBATCH --time=HH:MM:SS        # Without this, SLURM rejects the job
#SBATCH --mem=XG               # Without this, you get a tiny default that will crash your job
#SBATCH --output=path/file.out # Without this, you cannot see what went wrong
```

### Monitoring Jobs

```bash
# See your jobs (running and queued)
squeue -u your_x500

# See all jobs on a partition (to gauge wait times)
squeue -p agsmall

# Cancel a job
scancel 12345678

# See resource usage after a job completes
seff 12345678

# Watch live output while a job runs
tail -f logs/job_12345678.out
```

`seff` is the most useful post-run command — it shows how much memory and CPU time you actually used vs. what you requested, so you can tune future jobs.

### What You See in Your Terminal When You Submit Our Pipeline

When you run `bash submit_slsn_batch.slurm`, the submission script itself prints to your terminal before any job starts. Here is what to expect:

**1 — Pre-flight check (runs immediately in your terminal):**
```
============================================================
PRE-FLIGHT METRIC ALIGNMENT CHECK
============================================================
  Metrics instantiated in runners.py : ['SLSN_CharacterizeMetric', 'SLSN_Detect_Metric', ...]
  Metrics in _name_map               : ['SLSN_CharacterizeMetric', 'SLSN_Detect_Metric', ...]
  Output .npy short names            : ['characterize', 'detect', 'elasticc', 'spectrigger', 'villar']
  STATUS: OK - all 5 metrics aligned
```
This verifies that metrics in `runners.py` match the filename map before burning compute hours. If it fails, nothing is submitted.

**2 — Submission parameters:**
```
============================================================
SLSN BATCH LAUNCHER
============================================================
Repo      : /users/1/andra104/.../SLSNe_Metric
Partition : agsmall / cough052
Resources : 4 CPUs, 128G RAM, 24:00:00
Cadences  : baseline_v5.1.1_10yrs four_roll_v5.0.0_10yrs noroll_v5.0.0_10yrs poor_weather_v5.0.1_10yrs
Workers   : 4 (one per cadence)
============================================================
```
Read this before walking away — it confirms exactly what will run.

**3 — Job IDs as each model is submitted:**
```
--- Priority 1: fe_dependent ---
  [job 12345678] fe_dependent x all cadences

--- Priority 2: o_dependent ---
  [job 12345679] o_dependent x all cadences

--- Priority 3: naive ---
  [job 12345680] naive x all cadences
```

**4 — Final summary, then your prompt returns:**
```
============================================================
SUBMITTED
  fe_dependent : 12345678
  o_dependent  : 12345679
  naive        : 12345680

Monitor:
  squeue -u andra104
  tail -f output/SLSNe/logs/fe_dependent_JOBID.out
============================================================
```

Everything after this point goes into the `.out` / `.err` log files, not your terminal.

---

## Section 9: Reading Job Logs

When a SLURM job runs, print output goes to log files instead of your terminal.

```
output/SLSNe/logs/fe_dependent_JOBID.out   ← print statements, progress
output/SLSNe/logs/fe_dependent_JOBID.err   ← errors and warnings
```

Watch live:
```bash
tail -f output/SLSNe/logs/fe_dependent_12345678.out
```

### What a Healthy Log Looks Like

**1. Redshift → distance conversion:**
```
[INFO] z_min = 0.10000 -> d_min = 432.849... Mpc
[INFO] z_max = 2.00000 -> d_max = 5271.234... Mpc
```

**2. Cadence banner (4 total):**
```
============================================================
Running cadence: baseline_v5.1.1_10yrs
============================================================
```

**3. Metric configuration block** — prints every metric's parameters before the MAF run. Check this to catch wrong settings early without waiting for the full run.

**4. Per-metric results:**
```
  [baseline_v5.1.1_10yrs] SLSN_Detect_Metric: 84.23% (1849123/2195000)
  Saved: output/SLSNe/fe_dependent/metric_values_detect_..._260407_1423.npy
```

**5. File verification:**
```
  [baseline_v5.1.1_10yrs] Verified 5 .npy files (2195000 rows each): [...]
```
Three-level check: file exists + nonzero size + correct row count.

**6. Worker completion:**
```
  [DONE] baseline_v5.1.1_10yrs — 1849123 detections
  [DONE] noroll_v5.0.0_10yrs — 2011045 detections
```

### The Silent Failure Problem

> ⚠️ SLURM reports whether the *shell script* succeeded — not whether Python inside it succeeded. A job can show `COMPLETED` with exit code 0 while the Python pipeline crashed halfway through.

**Always verify after a job completes:**
```bash
# Do the output files exist and are they nonzero?
ls -lh output/SLSNe/fe_dependent/metric_values_*.npy

# Did the log end with [DONE] or [FAILED]?
tail -50 output/SLSNe/logs/fe_dependent_12345678.out

# Any Python errors?
grep -i "error\|failed\|traceback" output/SLSNe/logs/fe_dependent_12345678.err

# Did it run out of memory or time?
seff 12345678
```

### Common Log Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `[INFO] z_min = ...` | Population bounds set correctly | None — expected |
| `Verified 5 .npy files` | All outputs saved and checked | None — run succeeded |
| `[DONE] cadence — N detections` | Worker finished | None — expected |
| `[FAILED] cadence: ...` | That cadence crashed | Check `.err`, fix, rerun that cadence only |
| `RuntimeError: .npy verification failed` | File missing or wrong size | Disk issue or mid-run crash — rerun |
| `RuntimeError: N cadence(s) failed` | One or more parallel workers died | Partial results safe; rerun failed cadences |
| No output after cadence banner | Job hit time or memory limit | Check `seff`, increase `--time` or `--mem` |

---

## Section 10: Moving Files

### From Your Laptop to MSI

Run these commands **on your laptop** (not on MSI):

```bash
# Single file to a specific folder
scp ~/Downloads/fiducial_models.csv msi:~/Documents/Research/Transient_Metrics/SLSNe_Metric/data/

# Whole folder
scp -r ~/Downloads/my_folder/ msi:~/Documents/Research/

# Pull a file from MSI down to your laptop
scp msi:~/Documents/Research/SLSNe_Metric/output/summary.csv ~/Downloads/
```

> 💡 These use the `msi` SSH shortcut from Section 4c. If you haven't set that up yet, replace `msi:` with `your_x500@agate.msi.umn.edu:`.

### Downloading Directly to MSI from the Web

If you have a URL, skip your laptop entirely and download straight to MSI from the login node:

```bash
# Navigate to your destination first
cd ~/Documents/Research/Transient_Metrics/SLSNe_Metric/data/

# Single file
wget https://example.com/path/to/file.fits

# Whole folder (adjust --cut-dirs to match the number of path segments in your URL)
# e.g. https://website/1/2/3/4/5/6/7/8/ has 8 segments → --cut-dirs=8
wget -r -np -nH --cut-dirs=8 "https://website/1/2/3/4/5/6/7/8/"

# If restarting an interrupted download — skip files already downloaded
wget -r -np -nH --cut-dirs=8 -nc "https://website/1/2/3/4/5/6/7/8/"
```

Downloading files is light enough for the login node — no compute node needed.

**If the download will take a long time and you want to close your laptop:**
```bash
tmux new -s download
wget -r -np -nH --cut-dirs=8 "https://website/1/2/3/4/5/6/7/8/"
# Ctrl+B then D to detach — download keeps running
# tmux attach -t download to check on it later
```

Without `tmux`, closing your laptop drops the SSH connection and kills the download.

---

## Section 11: Navigating the File System

| Location | Path | What it's for |
|----------|------|---------------|
| Home directory | `/users/1/your_x500/` | Personal files, configs, your project repo |
| Shared project space | via `path_to_project` command | Shared lab data, large datasets |

To find a shared project path:
```bash
path_to_project PROJECT_NAME
```

For the SLSN project repo location on Agate, ask Cristy.

---

## Quick Reference Checklist

**First-time setup (do once):**
- [ ] Michael has added you to the MSI PI group
- [ ] Received MSI account confirmation email
- [ ] Cisco Secure Client (VPN) installed
- [ ] Tested OnDemand login at ondemand.msi.umn.edu
- [ ] SSH keys generated and copied to MSI
- [ ] `~/.ssh/config` shortcut set up (`ssh msi` works)
- [ ] Can locate the project repo on Agate

**Every session from home:**
1. Open Cisco Secure Client → `vpn.umn.edu` → Duo approve
2. Connect via OnDemand or `ssh msi`
3. When done: delete OnDemand sessions, disconnect VPN

**Before submitting a pipeline job:**
1. `git status` — are you on the right branch?
2. `mkdir -p output/SLSNe/logs`
3. `bash submit_slsn_batch.slurm` — read the pre-flight output before walking away
4. `squeue -u your_x500` — confirm jobs entered the queue

---

## Useful Links

| Resource | URL |
|----------|-----|
| Open OnDemand | https://ondemand.msi.umn.edu |
| MSI Getting Started | https://msi.umn.edu/getting-started |
| MSI Connecting to HPC | https://msi.umn.edu/getting-started/connecting/connecting-to-hpc-resources |
| MSI SSH Keys guide | https://msi.umn.edu/getting-started/getting-started-and-access/interactive-connections-faqs/how-do-i-setup-ssh-keys |
| UMN VPN | https://it.umn.edu/services-technologies/virtual-private-network-vpn |
| MSI Help Desk | help@msi.umn.edu |
| MSI System Status | https://msi.umn.edu (banner at top) |

---

*Last updated: April 2026. Check MSI documentation for changes to partition names or OnDemand interface.*
