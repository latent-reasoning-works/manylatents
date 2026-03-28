# Cluster Deploy Preflight Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a personal Claude Code skill that codifies the preflight-submit-monitor lifecycle for SLURM cluster deployments across Mila, DRAC/Narval, Tamia, and future clusters.

**Architecture:** A `~/.claude/skills/cluster-deploy-preflight/` directory with a SKILL.md (the skill logic), a `clusters.yaml` registry (per-cluster quirks), and a `deployment-history.md` log (learned outcomes). The skill is procedural — Claude reads the SKILL.md instructions and executes shell commands against the target cluster. No code beyond the skill definition files.

**Tech Stack:** Claude Code skill (markdown), YAML registry, ssh/rsync/squeue/sacct shell commands, Hydra cluster configs from manylatents.

---

## File Structure

| File | Purpose |
|------|---------|
| `~/.claude/skills/cluster-deploy-preflight/SKILL.md` | Main skill — preflight checklist, job sizing, submit, monitor, trigger logic |
| `~/.claude/skills/cluster-deploy-preflight/clusters.yaml` | Known cluster registry — connection details, quirks, constraints |
| `~/.claude/skills/cluster-deploy-preflight/deployment-history.md` | Deployment outcome log — what worked/failed, learned limits (Phase 3) |
| `~/.claude/skills/cluster-deploy-preflight/monitoring.md` | Monitoring reference — squeue patterns, log parsing, failure signatures (Phase 2) |

---

## Phase 1: Preflight + Submit

### Task 1: Create skill directory and SKILL.md skeleton

**Files:**
- Create: `~/.claude/skills/cluster-deploy-preflight/SKILL.md`

- [ ] **Step 1: Create skill directory**

```bash
mkdir -p ~/.claude/skills/cluster-deploy-preflight
```

- [ ] **Step 2: Write SKILL.md with frontmatter and section headers**

Write `~/.claude/skills/cluster-deploy-preflight/SKILL.md` with this content:

```markdown
---
name: cluster-deploy-preflight
description: Use when submitting SLURM jobs to any cluster (Mila, Narval, Tamia), when conversation involves cluster= commands, or when explicitly invoked for deployment preparation - runs preflight checks, job sizing, submit confirmation, and background monitoring
---

# Cluster Deploy Preflight

## Overview

Preflight-submit-monitor lifecycle for SLURM cluster deployments. Verifies tools, connectivity, constraints, and job sizing before submission, then monitors in the background.

**Core principle:** Never submit blind. Check the environment, size the job, confirm the queue, watch for early failures.

## Trigger Modes

This skill activates in three ways. Once preflight has completed (any trigger), do NOT re-trigger unless the user explicitly requests it.

**A) Explicit** — User invokes `/deploy-preflight <cluster>` or asks to prepare a cluster deployment.

**B) Automatic** — A `cluster=` argument appears in a SLURM submission command (e.g., `uv run python -m manylatents.main -m cluster=mila ...`). If preflight has not run this session, run it before proceeding with the submission.

**C) Session-start** — The conversation context involves cluster work (user mentions deploying, submitting, sweeps on a cluster). Load cluster context proactively.

## Preflight Checklist

Run these checks in order. Report results as a compact table. Stop on blockers (missing ssh, unreachable host).

### 1. Identify Target Cluster

Extract cluster name from:
- Explicit argument: `/deploy-preflight mila`
- Command parse: `cluster=narval` in a submission command
- Ask the user if ambiguous

Read the cluster registry (`clusters.yaml` in this skill directory) for known cluster details. If the cluster is unknown, ask the user for connection details and offer to save.

### 2. Tool Availability

Check these tools are on PATH or at known locations:

| Tool | Required For | Check Command |
|------|-------------|---------------|
| `ssh` | Connectivity | `which ssh` |
| `rsync` | Data staging (offline clusters) | `which rsync` |
| `squeue` | Job monitoring (on-cluster only) | `which squeue` |
| `sacct` | Job history (on-cluster only) | `which sacct` |
| `gh` | PR/CI checks | `which gh` |
| `uv` | Python env | `which uv` |

squeue/sacct may not be available when submitting remotely — that's OK, note it and skip.

### 3. Connectivity Verification

```bash
# SSH reachability (5s timeout)
ssh -o ConnectTimeout=5 -o BatchMode=yes <ssh_host> echo "ok" 2>&1
```

If this fails, report the error and stop. Common fixes:
- VPN not connected
- SSH key not loaded (`ssh-add`)
- Wrong hostname

### 4. Read Cluster Configs

Read the Hydra cluster config to surface constraints:

```bash
# Find the cluster config in the current project
cat <project_root>/manylatents/configs/cluster/<cluster_name>.yaml
cat <project_root>/manylatents/configs/resources/gpu.yaml  # if GPU job
cat <project_root>/manylatents/configs/resources/cpu.yaml  # if CPU job
```

Surface key info:
- Default partition (or "none — user must specify")
- Module loads required
- Default timeout, memory, CPUs
- Array parallelism setting

### 5. Job Sizing Guidance

Cross-reference the job parameters with known cluster limits:

**Mila CPU quota check:**
- `cpus_per_task × array_parallelism` = total concurrent CPUs
- Default GPU: 4 × 16 = 64 CPUs — hits QOSMaxCpuPerUserLimit for large sweeps
- Recommend: `cpus_per_task=2 array_parallelism=8` for sweeps > 30 jobs

**Narval partition check:**
- `partition: null` in config — user MUST specify (e.g., `hydra.launcher.partition=gpu`)
- Account auto-detection favors `rrg-*_gpu`

**General rules:**
- GPU jobs: check `gpus_per_task >= 1` (resources=gpu sets this)
- Memory: 32G default for GPU, 16G for CPU — flag if dataset is known to be large
- Timeout: flag if sweep has many configs and timeout < 1h

### 6. Data Staging (Offline Clusters Only)

If the registry marks `has_internet: false` and `rsync_target` is set:

```bash
# Rsync smoke test — small file round-trip
echo "preflight-test-$(date +%s)" > /tmp/preflight_smoke_test.txt
rsync -avz /tmp/preflight_smoke_test.txt <ssh_host>:<rsync_target>/preflight_smoke_test.txt
ssh <ssh_host> "cat <rsync_target>/preflight_smoke_test.txt && rm <rsync_target>/preflight_smoke_test.txt"
rm /tmp/preflight_smoke_test.txt
```

Report success/failure. If rsync fails, this is a blocker for offline clusters.

### 7. Preflight Summary

Print a compact summary table:

```
Cluster: mila (on-cluster)
SSH: OK
Tools: ssh ✓  rsync ✓  squeue ✓  gh ✓  uv ✓
Partition: main
Module loads: (none)
Resources: 4 CPUs, 1 GPU, 32G mem (resources=gpu)
Parallelism: 16 (64 CPUs concurrent)
⚠ Warning: 64 concurrent CPUs may hit quota for sweeps > 30 jobs
Internet: yes
Data staging: not required
```

## Submit Confirmation

After preflight passes:

1. Show the exact command that will be run
2. Highlight any overrides the user should consider (based on sizing guidance)
3. Wait for user confirmation
4. Run the submission command
5. Capture and report the SLURM job ID(s)

```bash
# After submission, verify job is queued
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" | head -20
```

## Job Monitoring

After submission, monitor in the background. Do NOT block the conversation.

### Poll Loop

Use `run_in_background` Bash commands. Check every 2-3 minutes.

```bash
# Check job state
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed,Timelimit --noheader
```

### State Transitions to Surface

| Transition | Action |
|------------|--------|
| PENDING → RUNNING | Brief note: "Job <id> running on <node>" |
| RUNNING → COMPLETED | Report: wall time, max memory, exit code. Read output scores if available. |
| RUNNING → FAILED | **Alert:** show exit code, read last 30 lines of .err file, suggest fix |
| RUNNING → OUT_OF_MEMORY | **Alert:** suggest `--mem=<2x current>` or reduce batch size |
| RUNNING → TIMEOUT | **Alert:** suggest longer `--time` or fewer configs per job |
| PENDING > 10 min | Note: "Job still pending — queue may be busy" |

### Failure Log Reading

```bash
# Find SLURM log files
ls -t logs/*/multiruns/*/$(date +%Y-%m-%d)/*/.submitit/*/stderr 2>/dev/null | head -5
# Or from sacct
sacct -j <job_id> --format=JobID,State,ExitCode --noheader
# Read the error log
tail -30 <stderr_path>
```

### Common Failure Signatures

| Pattern in stderr | Diagnosis | Fix |
|-------------------|-----------|-----|
| `slurmstepd: error: Detected 1 oom-kill` | Out of memory | Increase `--mem` |
| `ModuleNotFoundError` | Missing Python package | Check venv activation in setup commands |
| `module: command not found` | Module system not loaded | Add `module load` to setup |
| `CUDA error: out of memory` | GPU OOM | Reduce batch size or use larger GPU |
| `QOSMaxCpuPerUserLimit` | CPU quota exceeded | Reduce `cpus_per_task` or `array_parallelism` |
| `sbatch: error: Batch job submission failed` | SLURM rejection | Check partition, account, resource limits |

## Unknown Cluster Protocol

When the user targets a cluster not in the registry:

1. Ask: "I don't have <cluster> in my registry. Can you tell me:"
   - SSH hostname
   - Does it have internet access?
   - Any required module loads?
   - Default partition (or must specify)?
   - Rsync target for data staging (if offline)?
2. Run connectivity check with provided hostname
3. Offer: "Save this to the cluster registry for next time?"
4. If yes, append to `clusters.yaml`

## Deployment History

After each deployment completes (success or failure), append a one-line entry to `deployment-history.md`:

```markdown
| 2026-03-18 | mila | gpu | 16 jobs | COMPLETED | 2h13m | 28G peak | — |
| 2026-03-18 | narval | gpu | 60 jobs | FAILED | 0h02m | — | QOSMaxCpuPerUserLimit; fix: parallelism=8 |
```

Reference this history during future preflight sizing guidance.

## Common Mistakes

### Submitting without preflight on a new cluster
- **Problem:** Missing modules, wrong partition, no account → immediate failure
- **Fix:** Always run preflight. The auto-trigger (mode B) catches this.

### Ignoring quota warnings
- **Problem:** 60-job sweep with default parallelism hits CPU limit, all jobs stuck in PENDING
- **Fix:** Skill calculates total CPUs and warns before submission

### Forgetting module loads on DRAC
- **Problem:** `module: command not found` or wrong Python version
- **Fix:** Registry stores required module loads per cluster, preflight surfaces them

### Not monitoring after submit
- **Problem:** OOM failure 30 seconds in, but user doesn't notice for hours
- **Fix:** Background monitoring catches early failures immediately
```

- [ ] **Step 3: Verify skill file is readable**

```bash
head -5 ~/.claude/skills/cluster-deploy-preflight/SKILL.md
```

Expected: shows the YAML frontmatter with `name: cluster-deploy-preflight`.

- [ ] **Step 4: Commit**

This is a personal skill outside the repo — no git commit needed.

---

### Task 2: Create cluster registry

**Files:**
- Create: `~/.claude/skills/cluster-deploy-preflight/clusters.yaml`

- [ ] **Step 1: Write clusters.yaml with Mila, Narval, and Tamia entries**

```yaml
# Cluster Deploy Preflight — Known Clusters Registry
# Edit this file to add/update cluster details.
# The skill reads this at preflight time.

mila:
  ssh_host: login.server.mila.quebec
  has_internet: true
  module_loads: []
  default_partition: main
  partitions_available: [main, long, unkillable]
  quota_notes: >
    QOSMaxCpuPerUserLimit: default resources=gpu is 4 CPUs/task × 16 parallel = 64 CPUs.
    For sweeps > 30 jobs, use cpus_per_task=2 array_parallelism=8.
  rsync_target: null  # On-cluster, no rsync needed
  notes: >
    Primary cluster. Auto-routes to main-cpu partition when gpus_per_task=0.
    Use resources=gpu override to land on GPU nodes.
  hydra_config: manylatents/configs/cluster/mila.yaml

mila_remote:
  ssh_host: login.server.mila.quebec
  has_internet: true
  module_loads: []
  default_partition: main
  partitions_available: [main, long, unkillable]
  quota_notes: >
    Same as mila. Requires shop package (RemoteSlurmLauncher).
  rsync_target: null
  notes: >
    Remote submission via SSH from local machine. Requires ~/.ssh/id_rsa.
    Uses shop.hydra.launchers.RemoteSlurmLauncher.
  hydra_config: manylatents/configs/cluster/mila_remote.yaml

narval:
  ssh_host: narval.computecanada.ca
  has_internet: false
  module_loads: ["python/3.10"]
  default_partition: null  # User must specify
  partitions_available: []  # Varies by allocation
  quota_notes: >
    DRAC alliance. Account auto-detection favors rrg-*_gpu.
    No default partition — user must specify via hydra.launcher.partition=<partition>.
  rsync_target: "~/scratch/"
  notes: >
    Air-gapped (no internet). Must rsync data and wheels before submission.
    pip install from local wheelhouse only.
  hydra_config: manylatents/configs/cluster/narval.yaml

tamia:
  ssh_host: null  # TODO: fill in when first used
  has_internet: null  # TODO: determine
  module_loads: []  # TODO: determine
  default_partition: null  # TODO: determine
  partitions_available: []
  quota_notes: "Unknown — fill in after first deployment"
  rsync_target: null  # TODO: determine
  notes: "Placeholder entry. Will be populated on first use."
  hydra_config: null  # No Hydra config yet
```

- [ ] **Step 2: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('$HOME/.claude/skills/cluster-deploy-preflight/clusters.yaml')); print('valid')"
```

Expected: `valid`

---

### Task 3: Create deployment history log

**Files:**
- Create: `~/.claude/skills/cluster-deploy-preflight/deployment-history.md`

- [ ] **Step 1: Write initial deployment history file**

```markdown
# Deployment History

Outcomes from past cluster deployments. Referenced during preflight job sizing.

| Date | Cluster | Resources | Jobs | Status | Wall Time | Peak Mem | Notes |
|------|---------|-----------|------|--------|-----------|----------|-------|
```

---

### Task 4: Create monitoring reference

**Files:**
- Create: `~/.claude/skills/cluster-deploy-preflight/monitoring.md`

- [ ] **Step 1: Write monitoring reference with failure signatures and poll commands**

```markdown
# Monitoring Reference

## Poll Commands

### Check job state
```bash
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed,Timelimit --noheader
```

### Check queue position
```bash
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.6D %R" | head -20
```

### Read error logs
```bash
# Find most recent SLURM stderr files
ls -t logs/*/multiruns/$(date +%Y-%m-%d)/*/.submitit/*/stderr 2>/dev/null | head -5
# Read last 30 lines
tail -30 <stderr_path>
```

### Check resource usage after completion
```bash
sacct -j <job_id> --format=JobID,MaxRSS,MaxVMSize,AveRSS,Elapsed,State,ExitCode --noheader
```

## Failure Signatures

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| `slurmstepd: error: Detected 1 oom-kill` | Host OOM | Increase `--mem` (e.g., 32G → 64G) |
| `CUDA error: out of memory` | GPU OOM | Reduce batch size, use bigger GPU, or gradient checkpointing |
| `ModuleNotFoundError: No module named` | Missing package | Check venv activation, reinstall |
| `module: command not found` | Module system not loaded | Cluster setup commands missing `module load` |
| `QOSMaxCpuPerUserLimit` | CPU quota hit | Reduce `cpus_per_task` or `array_parallelism` |
| `sbatch: error: Batch job submission failed` | SLURM rejected | Check partition, account, resource limits |
| `DependencyNeverSatisfied` | Dependency job failed | Check parent job logs |
| `Exceeded job memory limit` | DRAC memory enforcement | Increase `--mem` in submission |
| `CANCELLED by` | Admin or user cancelled | Check if preempted (unkillable partition?) |
| `TimeLimit` | Wall time exceeded | Increase `--time` or split into smaller jobs |

## State Machine

```
PENDING → RUNNING → COMPLETED (success)
                  → FAILED (check exit code + stderr)
                  → OUT_OF_MEMORY (increase mem)
                  → TIMEOUT (increase time)
                  → CANCELLED (check reason)
```

## Poll Intervals

- First 5 minutes: check every 60 seconds (catch immediate failures)
- 5-30 minutes: check every 3 minutes
- After 30 minutes: check every 10 minutes
- Surface to user only on state transitions or anomalies
```

---

### Task 5: Smoke test the skill

- [ ] **Step 1: Verify skill is discoverable by Claude Code**

Start a new Claude Code session and check if the skill appears in the available skills list. The skill should show up with its description starting with "Use when submitting SLURM jobs..."

- [ ] **Step 2: Test explicit trigger on Mila**

Invoke the skill explicitly:
- It should read `clusters.yaml`, find the Mila entry
- Run tool checks (ssh, rsync, squeue, gh, uv)
- Check SSH connectivity to `login.server.mila.quebec`
- Read the Hydra cluster config
- Print the preflight summary table

- [ ] **Step 3: Test with a real dry-run submission**

Run a manylatents smoke test with `--cfg job` (dry run, no actual submission) to verify the skill correctly parses the command and surfaces sizing guidance:

```bash
uv run python -m manylatents.main --cfg job cluster=mila resources=gpu algorithms/latent=pca data=swissroll
```

- [ ] **Step 4: Test unknown cluster path**

Invoke with a cluster not in the registry (e.g., `cedar`) and verify the skill asks for connection details and offers to save.

- [ ] **Step 5: If any issues found, fix and re-test**

---

## Phase 2: Monitoring (Future)

### Task 6: Add background monitoring to SKILL.md

**Files:**
- Modify: `~/.claude/skills/cluster-deploy-preflight/SKILL.md` (monitoring section already drafted — wire up poll loop details)

- [ ] **Step 1: Test squeue/sacct availability on Mila login node**
- [ ] **Step 2: Test background poll with a real short job** (e.g., `fast_dev_run=true`)
- [ ] **Step 3: Verify state transition surfacing works** (PENDING → RUNNING → COMPLETED)
- [ ] **Step 4: Test failure detection** (submit a job that will OOM or timeout intentionally)
- [ ] **Step 5: Update monitoring.md with any new failure signatures discovered**

---

## Phase 3: Multi-Cluster Intelligence (Future)

### Task 7: Deployment history tracking

- [ ] **Step 1: After a real deployment, manually append an entry to deployment-history.md**
- [ ] **Step 2: Verify the skill references the history during subsequent preflight sizing**
- [ ] **Step 3: Test with Narval** — submit a job, record outcome, verify next preflight uses learned data

### Task 8: Port forwarding recipes

- [ ] **Step 1: Add port forwarding section to clusters.yaml** (per-cluster SSH tunnel commands for WandB, Jupyter, TensorBoard)
- [ ] **Step 2: Add instructions to SKILL.md for when/how to offer port forwarding setup**

### Task 9: Cross-cluster validation

- [ ] **Step 1: First real deployment to Narval** — fill in any missing registry details
- [ ] **Step 2: First real deployment to Tamia** — fill in placeholder entry
- [ ] **Step 3: Update clusters.yaml with learned details from each cluster**
