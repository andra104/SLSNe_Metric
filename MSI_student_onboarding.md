# MSI Access Guide for New Students
### Minnesota Supercomputing Institute (Agate Cluster)
*Prepared for onboarding to the Miller Lab / SLSN pipeline project*

---

## Step 0: Before You Can Log In

**Your advisor (Michael) must add you to the MSI PI group first.**
MSI organizes access by PI group — you cannot log in until your faculty advisor has added your UMN Internet ID to their group via the MyMSI portal. Remind Michael to do this before you try anything below.

Once added, you will receive a confirmation email. Your MSI username is your UMN Internet ID (x500), e.g., `smith123`.

---

## Step 1: Are You On Campus or Off?

This determines what you need to do before connecting.

| Location | Network | What you need |
|----------|---------|---------------|
| On campus | eduroam WiFi | Nothing extra — you're already on the UMN network |
| Off campus | Home/other WiFi | **Must connect to UMN VPN first** (see Section 2) |

> ⚠️ MSI will silently refuse connections from outside the UMN network — no error message, it just won't connect. Always check that VPN is on when working from home.

---

## Section 2: Connecting from Off Campus (VPN Setup)

You need the **Cisco Secure Client** (formerly called AnyConnect). This is a UMN-provided VPN app.

### Download and Install

1. Go to: **[it.umn.edu/vpn](https://it.umn.edu/services-technologies/virtual-private-network-vpn)**
2. Download the Cisco Secure Client installer for your OS (Mac, Windows, or Linux)
3. Install it — it will appear as "Cisco Secure Client" in your applications

### Connect to VPN

1. Open **Cisco Secure Client**
2. In the connection field, select or type: **`vpn.umn.edu`**
3. Choose the tunnel type when prompted:
   - **"UMN - Split Tunnel - General Access"** — use this for most things (including MSI)
4. Enter your UMN Internet ID and password
5. **Approve the Duo push** on your phone
6. The status bar will say **"Connected"**

> 🔔 Sessions auto-disconnect after 60 minutes of idle time, and after 7 days total. If MSI suddenly stops responding, check that your VPN is still connected.

### Disconnect When Done
Click the VPN icon in your menu bar/taskbar → **Disconnect**. Don't leave it running unnecessarily.

---

## Section 3: Two Ways to Access MSI

Once you're on the UMN network (on campus or via VPN), you have two options:

| Method | Best for | Interface |
|--------|----------|-----------|
| **Open OnDemand** (browser) | Beginners, Jupyter notebooks, visual work | Web browser |
| **SSH terminal** | Power users, running SLURM jobs, scripting | Terminal / command line |

Both connect to the same cluster (Agate) and the same files. They are different doors into the same house.

---

## Section 4: Open OnDemand (Browser Access)

This is the easiest way to get started. No software to install beyond your browser.

### Log In

1. Go to: **[ondemand.msi.umn.edu](https://ondemand.msi.umn.edu)**
2. Log in with your UMN Internet ID and password
3. Complete Duo authentication

### What You'll See

The OnDemand dashboard has a menu bar at the top with:
- **Files** — browse and upload/download files on MSI
- **Jobs** — view your running/queued SLURM jobs
- **Interactive Apps** — launch Jupyter, Desktop sessions, etc.

### Option A: Start a Jupyter Notebook Session

1. Click **Interactive Apps → Jupyter**
2. Fill in the resource form:
   - **Cluster:** Agate
   - **Account:** `cough052` (this is the lab's account)
   - **Partition:** `agsmall`
   - **Number of hours:** however long you need (start with 2)
   - **Number of cores:** 1–2 for most notebook work
   - **Memory:** 8 GB is usually enough
3. Click **Launch**
4. Wait for the session card to show **"Running"** (usually < 2 min)
5. Click **Connect to Jupyter**

> ✅ This opens JupyterLab in your browser. You are now running code on an MSI compute node, not your laptop.

### Option B: Start a Desktop Session

1. Click **Interactive Apps → Desktop**
2. Fill in the resource form (same fields as above)
3. Click **Launch**, then **Launch Desktop**
4. A Linux desktop opens in your browser — you can open a terminal from here

> 💡 The Desktop is like having a Linux laptop inside your browser. It's useful if you need a graphical file manager or want to run terminal commands without setting up SSH.

### OnDemand Terminal (Quick Access)

From the OnDemand menu: **Clusters → Agate Shell Access**

This opens a terminal directly in your browser — it lands you on a **login node** (not a compute node). Good for quick file checks, navigating directories, and submitting SLURM jobs. **Do not run heavy computations here.**

### End Your Session

When done: go back to **My Interactive Sessions** and click **Delete** on your session card. This releases compute resources for others. Don't just close the browser tab.

---

## Section 5: SSH Access (Terminal on Your Own Computer)

SSH lets you connect to MSI directly from a terminal on your laptop. This is what most researchers use for submitting SLURM batch jobs and scripting.

### Mac / Linux

Mac and Linux have a built-in Terminal app. On Mac, search for "Terminal" in Spotlight.

**To connect:**
```bash
ssh your_x500@agate.msi.umn.edu
```
Replace `your_x500` with your UMN Internet ID (e.g., `smith123`).

You'll be prompted for:
1. Your MSI/UMN password
2. Duo authentication (approve the push on your phone)

You land on a **login node**. This is for navigation and job submission, not heavy computation.

### Windows

Windows 10 and 11 have SSH available in PowerShell/Command Prompt:
```
ssh your_x500@agate.msi.umn.edu
```

Alternatively, download **PuTTY** (free, lightweight):
1. Download from [putty.org](https://www.putty.org)
2. Open PuTTY → enter `agate.msi.umn.edu` as the Host Name
3. Connection type: SSH
4. Click Open → enter your username and password

### Set Up SSH Keys (Recommended — Saves You from Typing Password Every Time)

SSH keys let you connect without entering your password each time.

**On Mac/Linux:**
```bash
# 1. Generate a key pair on your laptop (run this locally, not on MSI)
ssh-keygen -t ed25519 -C "your_x500@umn.edu"
# Press Enter to accept defaults (no passphrase needed for convenience)

# 2. Copy your public key to MSI
ssh-copy-id your_x500@agate.msi.umn.edu
# Enter your password one last time

# 3. Future logins (no password needed):
ssh your_x500@agate.msi.umn.edu
```

**On Windows:** Use PuTTYgen to create a key pair, then paste the public key into `~/.ssh/authorized_keys` on MSI (see [MSI's SSH key guide](https://msi.umn.edu/getting-started/getting-started-and-access/interactive-connections-faqs/how-do-i-setup-ssh-keys)).

---

## Section 6: Login Nodes vs. Compute Nodes — Know the Difference

If you've never worked on a computing cluster before, the way MSI is organized can be confusing at first. This section explains the mental model you need before you start running anything.

---

### The Analogy: A Restaurant Kitchen

Think of MSI like a restaurant:

- **You** are a customer placing an order
- **The login node** is the front desk — where you walk in, look at the menu, place your order
- **SLURM** is the order ticket system — it takes your order and routes it to the right station
- **The compute nodes** are the kitchen — where the actual cooking (computation) happens

You don't walk into the kitchen and start cooking yourself. You place your order at the front, and the kitchen handles it. That's exactly how MSI works.

---

### What a Login Node Is

When you SSH into MSI — whether from a terminal, VS Code, or OnDemand's shell — you land on a **login node**. This is a shared machine that everyone at MSI is also logged into at the same time.

**The login node is for:**
- Navigating files (`cd`, `ls`, `cat`)
- Editing code (`nano`, `vim`, or through VS Code)
- `git` commands (pull, push, commit)
- Writing and submitting SLURM job scripts
- Checking job status (`squeue`, `scancel`)
- Light, fast commands that finish in seconds

**The login node is NOT for:**
- Running a Python script that takes more than a minute or two
- Running a Jupyter notebook with heavy computations
- Processing large datasets directly
- Anything that uses significant CPU or memory for an extended time

Because it's shared, running heavy code on the login node slows it down for everyone on the cluster simultaneously. MSI will sometimes kill jobs that abuse the login node.

---

### What SLURM Is

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduler. Its job is to:

1. Accept your request for compute resources ("I need 4 cores, 16 GB of RAM, for 2 hours")
2. Put you in a queue with everyone else's requests
3. When resources are available, run your code on a compute node
4. Send you an email or log when it's done

SLURM is **not** the computation itself — it's the system that *routes* your computation to the right place.

---

### What a Compute Node Is

A **compute node** is a dedicated machine (or one of many machines) whose entire job is to run computations. Unlike the login node, a compute node is allocated *just to you* for the time you requested. You don't share it with other users during your job.

MSI's Agate cluster has many compute nodes — when you submit a SLURM job, SLURM picks one that's free and runs your code there.

---

### Submitting a SLURM Job: How It Actually Works

You write a **job script** — a regular shell script with special SLURM instructions at the top — and submit it with `sbatch`. Here's a minimal example:

```bash
#!/bin/bash
#SBATCH --job-name=my_test_job       # name shown in the queue
#SBATCH --account=cough052           # the lab's MSI account
#SBATCH --partition=agsmall          # which partition (queue) to use
#SBATCH --time=01:00:00              # max wall time: 1 hour
#SBATCH --ntasks=1                   # number of tasks (usually 1)
#SBATCH --cpus-per-task=2            # cores per task
#SBATCH --mem=8gb                    # memory requested
#SBATCH --mail-type=END,FAIL         # email when job ends or fails
#SBATCH --mail-user=your_x500@umn.edu

# Everything below here is your actual code
conda activate rubin_sim_2.6.1
cd /path/to/your/project
python run_slsn_pipeline.py --some-flag
```

Submit it from the login node:
```bash
sbatch my_job_script.sh
```

SLURM responds immediately with a job ID:
```
Submitted batch job 12345678
```

Your terminal is free again instantly. The computation is now running in the background on a compute node. You can close your laptop and it will keep running.

---

### Checking and Managing Your Jobs

```bash
# See all your currently running or queued jobs
squeue -u your_x500

# Cancel a job (use the job ID from squeue output)
scancel 12345678

# See details about a completed job (resources used, exit code)
seff 12345678
```

---

### The Key Distinction: Submitting vs. Running

This is the thing that confuses most beginners:

| Action | Where it happens | Is it "heavy computation"? |
|--------|-----------------|---------------------------|
| Typing `sbatch my_job.sh` | Login node | ❌ No — this is just sending a message |
| Editing your Python file in VS Code | Login node | ❌ No — text editing is trivial |
| `git pull`, `git commit` | Login node | ❌ No — fast file operations |
| The Python code *inside* your job script | Compute node (SLURM moves it there) | ✅ Yes — this is the computation |
| Running `python run_pipeline.py` directly in your terminal | Login node | ❌ Don't do this |

**Submitting a job from a login node is always correct and expected.** The login node is never burdened by the computation itself — SLURM physically moves that work to a compute node.

---

### VS Code + SSH Users: You Are Not Causing a Slowdown

If you connect to MSI through VS Code's SSH extension and use the integrated terminal to submit SLURM jobs — **this is completely fine.** You are on the login node, but all you're doing there is typing `sbatch`. The actual computation goes elsewhere.

The only thing to avoid: running a heavy script directly in that VS Code terminal without going through SLURM (e.g., accidentally running `python run_slsn_pipeline.py` and letting it sit for 30 minutes). Use SLURM for anything that takes real time.

---

### Common Beginner Mistakes

| Mistake | What to do instead |
|---------|-------------------|
| Running a long Python script directly on the login node | Write a job script and use `sbatch` |
| Forgetting `--account=cough052` in the job script | SLURM will reject or mischarge the job — always include it |
| Requesting too little time and having the job killed | Request more time than you think you need; unused time is released |
| Closing your terminal thinking it killed the job | SLURM jobs keep running regardless of your connection — use `scancel` to actually stop one |
| Not checking `squeue` after submitting | Always verify your job actually entered the queue |

---

### Section 6b: Getting to the Nodes 

This is the most important concept to understand.

| Node Type | What it's for | How you get there |
|-----------|---------------|-------------------|
| **Login node** | Navigation, editing files, submitting jobs | SSH / OnDemand Shell |
| **Compute node** | Running actual computations | SLURM batch job OR OnDemand session |

**Never run heavy computation on a login node.** It's a shared resource — running a Python script that takes 10+ minutes there will affect everyone.

### Getting a Compute Node Interactively (via SSH)

If you're on a login node via SSH and need to run something quick interactively:
```bash
srun -N 1 -n 1 -t 2:00:00 -p agsmall --account=cough052 --pty bash
```
This requests 1 core for 2 hours on the `agsmall` partition under the lab account. Once the prompt changes (you'll see a new hostname like `acn001`), you're on a compute node.

### Submitting a SLURM Batch Job

For longer runs (like the SLSN pipeline), you submit a job script and it runs in the background:
```bash
sbatch your_job_script.sh
```
Check job status with:
```bash
squeue -u your_x500
```

---

## Section 7: Submitting Jobs in Practice — OnDemand Form vs. SLURM Script

> This section expands on Section 6. Both methods request the same resources from the same cluster — they're just different interfaces to SLURM.

There are two ways to submit a SLURM job:

| Method | Best for | How you launch it |
|--------|----------|-------------------|
| **OnDemand form** (browser) | Interactive sessions — Jupyter, Desktop, notebooks | Fill out a form, click Launch |
| **`.slurm` script** (`sbatch`) | Batch jobs — long pipeline runs, unattended computation | Write a script, run `sbatch` from terminal |

Both talk to SLURM. Both request compute nodes. The difference is just how you tell SLURM what you want.

---

### Part A: The OnDemand Form

When you launch a Desktop or Jupyter session through OnDemand, you fill out a resource request form. Here's what each field means, using the values from the screenshot above as a concrete example:

| Field | What it means | Example value | Notes |
|-------|--------------|---------------|-------|
| **Account** | Which lab account to charge compute time to | `cough052` | Always use this for our project |
| **Partition** | Which queue/pool of nodes to draw from | `agsmall` or `interactive` | `agsmall` for most work; `interactive` for short exploratory sessions |
| **Number of Nodes** | How many separate machines | `1` | Almost always 1 — you'd only use more for MPI-parallel work |
| **Cores per Node** | CPU cores on that machine | `4` | More cores = faster parallel code, but more resources consumed |
| **Memory per Node** | RAM allocated to your session | `32G` | Request what you need; unused memory is still held by you |
| **Scratch per Node** | Temporary fast local disk | `0` | Leave 0 unless your job needs local scratch space |
| **GPUs per Node** | GPU allocation | `0` | 0 unless running GPU code (e.g., deep learning) |
| **Time Limit** | How long your session runs before SLURM kills it | `8 Hours` | Session ends when time is up — save your work |

> 💡 **"Shorter times will probably start faster"** — that note on the form is real. If you request 1 hour instead of 8, SLURM can often find you a node immediately. Request only what you need.

Once you click **Launch**, OnDemand submits this as a SLURM job behind the scenes, waits for a node to be allocated, then opens your Desktop or Jupyter in the browser. You don't write any code to make this happen — the form does it for you.

**When your time limit is up:** the session ends automatically. You'll lose any unsaved notebook state. For long notebook runs, use `sbatch` instead (see Part B).

---

### Part B: The `.slurm` Script

For jobs that run unattended — like a full pipeline run that takes 13+ hours — you write a shell script with SLURM directives at the top and submit it with `sbatch`. This is what the OnDemand form is doing internally, just exposed as text you control.

#### Minimal Example

Here is the simplest possible SLURM script. Every field maps directly to something you'd fill in on the OnDemand form:

```bash
#!/bin/bash
# -------------------------------------------------------
# These lines are SLURM directives — they start with #SBATCH
# SLURM reads them; bash ignores them (they look like comments)
# -------------------------------------------------------
#SBATCH --job-name=my_test          # Label shown in squeue
#SBATCH --account=cough052          # Lab account (always required)
#SBATCH --partition=agsmall         # Which queue to use
#SBATCH --nodes=1                   # Number of machines (almost always 1)
#SBATCH --ntasks=1                  # Number of parallel tasks
#SBATCH --cpus-per-task=4           # Cores per task (= "Cores per Node" in form)
#SBATCH --mem=32G                   # RAM (= "Memory per Node" in form)
#SBATCH --time=08:00:00             # Wall time limit: HH:MM:SS
#SBATCH --output=logs/job_%j.out    # Where stdout goes (%j = job ID)
#SBATCH --error=logs/job_%j.err     # Where stderr goes
#SBATCH --mail-type=END,FAIL        # Email when job ends or fails
#SBATCH --mail-user=your_x500@umn.edu

# -------------------------------------------------------
# Everything below here is regular bash — your actual job
# -------------------------------------------------------

# Activate your conda environment
source /common/software/install/migrated/anaconda/miniconda3_4.8.3-jupyter/etc/profile.d/conda.sh
conda activate rubin_sim_2.6.1

# Move to your project directory
cd /users/1/your_x500/Documents/Research/YourProject

# Run your code
python run_my_script.py --some-flag
```

Submit it from any terminal on the login node:
```bash
sbatch my_job.sh
```

SLURM responds immediately:
```
Submitted batch job 12345678
```

Your terminal is free. The job is running in the background on a compute node. You can close VS Code, close your laptop, go home — it keeps running.

#### How the OnDemand Form Maps to Script Directives

| OnDemand Form Field | `.slurm` Script Equivalent |
|--------------------|---------------------------|
| Account | `#SBATCH --account=cough052` |
| Partition | `#SBATCH --partition=agsmall` |
| Number of Nodes | `#SBATCH --nodes=1` |
| Cores per Node | `#SBATCH --cpus-per-task=4` |
| Memory per Node | `#SBATCH --mem=32G` |
| Time Limit | `#SBATCH --time=08:00:00` |
| Email notification | `#SBATCH --mail-type=END,FAIL` |
| *(no form equivalent)* | `#SBATCH --output=logs/job_%j.out` — you must set this manually in scripts |

---

### Part C: What Our Real Job Script Looks Like

The minimal example above is stripped down for clarity. Here is what a real production SLURM script looks like for this project — `submit_slsn_batch.slurm`. You don't need to understand every line right away, but it's useful to see how the same concepts scale up.

```bash
#!/bin/bash
#SBATCH --job-name=slsn_fe          # Short name (truncated to 4 chars of model name)
#SBATCH --account=cough052
#SBATCH --partition=agsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # 4 cores — one per cadence running in parallel
#SBATCH --mem=128G                  # 4 workers x ~600MB population + MAF overhead
#SBATCH --time=24:00:00             # Up to 24 hours for a full model x 4 cadences
#SBATCH --output=output/SLSNe/logs/fe_dependent_%j.out
#SBATCH --error=output/SLSNe/logs/fe_dependent_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andra104@umn.edu
```

Key things to notice compared to the minimal example:

- **Memory is 128G instead of 32G** — because this job loads a population of ~2.2 million simulated supernovae into RAM across 4 parallel workers. More workers = more RAM needed.
- **Time is 24:00:00** — a full model run takes ~13 hours. The extra buffer prevents SLURM from killing the job before it finishes.
- **4 CPUs** — the script runs all 4 survey cadences simultaneously (one per core), so what would take ~40 hours sequentially finishes in ~13 hours.
- **The script submits 3 separate jobs** (one per rate model: `fe_dependent`, `o_dependent`, `naive`) — each gets its own job ID, runs independently, and emails when done.

The actual pipeline call inside the script looks like:
```bash
python3 run_slsn_pipeline.py \
    --model fe_dependent \
    --cadences baseline_v5.1.1_10yrs four_roll_v5.0.0_10yrs noroll_v5.0.0_10yrs poor_weather_v5.0.1_10yrs \
    --templates-pkl output/SLSNe/shared/templates.pkl \
    --n-cores 4 \
    --n-workers 4
```

This is the "heavy computation" — it runs on the compute node SLURM allocated. You never run this line directly in your terminal.

---

### Things That Must Always Be in a SLURM Script

These are the fields you cannot skip — SLURM will either reject the job or charge the wrong account:

```bash
#SBATCH --account=cough052     # REQUIRED — always. Without this, job may fail or charge wrong group.
#SBATCH --partition=agsmall    # REQUIRED — tells SLURM which pool of nodes to use.
#SBATCH --time=HH:MM:SS        # REQUIRED — SLURM needs a wall time limit.
#SBATCH --mem=XG               # REQUIRED — without this, you get a tiny default that may crash your job.
#SBATCH --output=path/file.out # Strongly recommended — otherwise you can't see what went wrong.
```

And always make your log directory before submitting:
```bash
mkdir -p output/SLSNe/logs
sbatch submit_slsn_batch.slurm
```

---

### Monitoring and Controlling Your Job After Submission

```bash
# See your jobs in the queue
squeue -u your_x500

# See everyone's jobs on a partition (to gauge wait times)
squeue -p agsmall

# Cancel a job
scancel 12345678

# See resource usage after a job completes (was 128G enough? did it time out?)
seff 12345678

# Watch live output as the job runs
tail -f output/SLSNe/logs/fe_dependent_12345678.out
```

The `seff` command is particularly useful — it tells you how much memory and CPU time the job actually used vs. what you requested, so you can tune future submissions.

---
### Applied Example: What You See in Your Terminal When You Submit

When you run `bash submit_slsn_batch.slurm` from your terminal (or VS Code), the script prints to your screen *before* any job starts. This is the submission script talking — not the pipeline. Here is what to expect:

**Step 1 — Pre-flight check runs immediately:**
```
============================================================
PRE-FLIGHT METRIC ALIGNMENT CHECK
============================================================
  Metrics instantiated in runners.py : ['SLSN_CharacterizeMetric', 'SLSN_Detect_Metric', ...]
  Metrics in _name_map               : ['SLSN_CharacterizeMetric', 'SLSN_Detect_Metric', ...]
  Output .npy short names            : ['characterize', 'detect', 'elasticc', 'spectrigger', 'villar']
  STATUS: OK - all 5 metrics aligned
```
This Python check runs right in your terminal before anything is submitted. It verifies that the metrics defined in `runners.py` match the filename map — catching mismatches before burning 13+ hours of compute. If it fails, the script stops and nothing is submitted.

**Step 2 — Submission parameters printed:**
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
Read this before walking away — it confirms the exact resources being requested and which cadences will run.

**Step 3 — Each job gets submitted and prints its job ID:**
```
--- Priority 1: fe_dependent ---
  [job 12345678] fe_dependent x all cadences

--- Priority 2: o_dependent ---
  [job 12345679] o_dependent x all cadences

--- Priority 3: naive ---
  [job 12345680] naive x all cadences
```

**Step 4 — Final summary and your terminal is free:**
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

Your prompt returns immediately after this. The jobs are now in SLURM's queue. Everything that happens next — the actual pipeline output — goes into the `.out` log files, not your terminal. See Section 8 for what to expect there.

---

## Section 8: Reading Your Job's Output Log

When a SLURM job runs, everything that would normally print to your terminal instead goes into a log file. For this project, logs are saved to:

```
output/SLSNe/logs/fe_dependent_JOBID.out   ← standard output (print statements)
output/SLSNe/logs/fe_dependent_JOBID.err   ← errors and warnings
```

Watch a job live while it's running:
```bash
tail -f output/SLSNe/logs/fe_dependent_12345678.out
```

---

### What a Healthy Log Looks Like

The pipeline prints structured progress messages at each stage. Here is what you should see in a successful run, in order:

**1. Redshift → distance conversion (top of log)**
```
[INFO] z_min = 0.10000 -> d_min = 432.849... Mpc
[INFO] z_max = 2.00000 -> d_max = 5271.234... Mpc
```
This confirms the population's distance bounds were set correctly.

**2. Cadence banner (once per cadence, 4 total)**
```
============================================================
Running cadence: baseline_v5.1.1_10yrs
============================================================
```

**3. Metric configuration block**
```
[baseline_v5.1.1_10yrs] --- Metric configuration ---
  SLSN_Detect_Metric: {'mag_limit': 24.5, 'use_extinction': True, ...}
  SLSN_SpecTriggerMetric: {'mag_limit': 23.0, 'peak_window': 30, ...}
----------------------------
```
This prints every metric's parameters before the MAF run starts. If a setting is wrong, you can catch it here without waiting for the full run to finish.

**4. Per-metric efficiency results**
```
  [baseline_v5.1.1_10yrs] SLSN_Detect_Metric: 84.23% (1849123/2195000)
  [baseline_v5.1.1_10yrs] SLSN_CharacterizeMetric: 61.07% (1340211/2195000)
  ...
  Saved: output/SLSNe/fe_dependent/metric_values_detect_fe_dependent_baseline_v5.1.1_10yrs_z0.1-2.0_260407_1423.npy
```

**5. File verification**
```
  [baseline_v5.1.1_10yrs] Verified 5 .npy files (2195000 rows each): ['characterize', 'detect', 'elasticc', 'spectrigger', 'villar']
```
This is a three-level check: file exists + nonzero size + correct number of rows. If any file fails this check, the pipeline raises an error immediately.

**6. Parallel worker completion (once per cadence)**
```
  [DONE] baseline_v5.1.1_10yrs — 1849123 detections
  [DONE] noroll_v5.0.0_10yrs — 2011045 detections
  ...
```

A healthy full run ends with all 4 cadences showing `[DONE]` and a final combined summary CSV written to disk.

---

### What a Failed Log Looks Like

**A cadence worker crashed:**
```
  [FAILED] noroll_v5.0.0_10yrs: RuntimeError: [noroll_v5.0.0_10yrs] .npy verification failed:
    spectrigger: missing
    run_tag: fe_dependent_noroll_v5.0.0_10yrs_z0.1-2.0_260407_1423
```
The pipeline tells you exactly which cadence failed, what kind of error it was, and which file is missing. The other cadences that completed successfully are not affected — their `.npy` files and per-cadence CSVs are already saved.

**A file was written but is empty:**
```
  [FAILED] baseline_v5.1.1_10yrs: RuntimeError: detect: zero bytes
```

---

### The Silent Failure Problem — Read This

> ⚠️ This is the most important thing to understand about SLURM jobs.

SLURM reports whether your *job script* succeeded — not whether the Python code inside it succeeded. A job can show `COMPLETED` with exit code `0` in `squeue` or via email, while the Python pipeline actually crashed halfway through.

**Always verify your outputs after a job completes:**

```bash
# Check that the expected .npy files exist and are nonzero
ls -lh output/SLSNe/fe_dependent/metric_values_*.npy

# Check the bottom of the log for [DONE] or [FAILED]
tail -50 output/SLSNe/logs/fe_dependent_12345678.out

# Check if Python exited with an error
grep -i "error\|failed\|traceback" output/SLSNe/logs/fe_dependent_12345678.err

# Check actual resource usage (did it run out of memory or time?)
seff 12345678
```

If `seff` shows `Memory Efficiency: 98%` or `State: OUT_OF_MEMORY`, that's why the job failed — request more memory next time.

---

### Common Log Messages and What They Mean

| Message | Meaning | Action |
|---------|---------|--------|
| `[INFO] z_min = ...` | Population bounds set correctly | None — this is expected |
| `Verified 5 .npy files` | All outputs saved and checked | None — run succeeded |
| `[DONE] cadence — N detections` | That cadence's worker finished | None — expected |
| `[FAILED] cadence: ...` | That cadence crashed | Check `.err` file, fix and rerun that cadence only |
| `RuntimeError: .npy verification failed` | File missing or wrong size after save | Disk issue or crash mid-write — rerun |
| `RuntimeError: N cadence(s) failed` | One or more parallel workers died | Partial results are safe; rerun failed cadences |
| No output after cadence banner | Job ran out of time or memory mid-run | Check `seff`, increase `--time` or `--mem` |

---

## Section 9: Navigating the File System

Your files live in two main places on MSI:

| Location | Path | What it's for |
|----------|------|---------------|
| Home directory | `/home/YOUR_X500/` | Personal files, small configs |
| Project space | `/home/PIGROUP/PROJECT/` or via `path_to_project` | Shared lab data, large files, pipeline outputs |

To find the path to a project:
```bash
path_to_project PROJECT_NAME
```

For the SLSN project specifically, ask Cristy for the exact project path and repo location on Agate.

---

## Quick Reference Checklist

**First-time setup:**
- [ ] Michael has added you to the MSI PI group
- [ ] You received MSI account confirmation
- [ ] Cisco Secure Client (VPN) installed
- [ ] Tested OnDemand login at ondemand.msi.umn.edu
- [ ] SSH key set up (optional but recommended)
- [ ] Can locate the project repo on Agate

**Every session from home:**
1. Connect Cisco Secure Client → `vpn.umn.edu` → Duo approve
2. Open OnDemand or SSH
3. When done: delete your OnDemand session and disconnect VPN

---

## Useful Links

| Resource | URL |
|----------|-----|
| Open OnDemand | https://ondemand.msi.umn.edu |
| MSI Getting Started | https://msi.umn.edu/getting-started |
| MSI Connecting to HPC | https://msi.umn.edu/getting-started/connecting/connecting-to-hpc-resources |
| MSI SSH Keys | https://msi.umn.edu/getting-started/getting-started-and-access/interactive-connections-faqs/how-do-i-setup-ssh-keys |
| UMN VPN (IT) | https://it.umn.edu/services-technologies/virtual-private-network-vpn |
| MSI Help Desk | help@msi.umn.edu |
| MSI System Status | https://msi.umn.edu (banner at top) |

---

*Last updated: April 2026. Check MSI documentation for any changes to partition names or OnDemand interface.*
