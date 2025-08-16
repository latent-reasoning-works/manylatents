import hydra
from hydra import initialize, compose
import sys
from pathlib import Path
import uuid
import submitit

def main():
    # Agent/user calls this script like: `python run_pipeline.py experiment/pipeline=pca_then_ae`
    overrides = sys.argv[1:]
    with initialize(config_path="manylatents/configs", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    pipeline_steps = cfg.pipeline
    
    run_name = cfg.get('name', 'pipeline_run')
    shared_fs_path = Path(f"/network/scratch/c/cesar.valdez/manyLatents/pipeline_runs/{run_name}-{uuid.uuid4().hex[:8]}")
    shared_fs_path.mkdir(parents=True, exist_ok=True)
    previous_job = None

    print(f"Orchestrating pipeline '{run_name}' with steps: {pipeline_steps}")

    for i, algo_name in enumerate(pipeline_steps):
        # Determine the full algorithm config path
        # Extract algorithm type and name from pipeline step
        if isinstance(algo_name, str):
            algo_path = algo_name
        else:
            # Handle dictionary format like {latent/pca: {param: value}}
            algo_path = list(algo_name.keys())[0] if isinstance(algo_name, dict) else str(algo_name)
        
        # Compose the full config for THIS step to discover its launcher settings
        with initialize(config_path="manylatents/configs", version_base=None):
            step_cfg = compose(config_name="config", overrides=[f"algorithm={algo_path}"])
        
        launcher_cfg = step_cfg.hydra.launcher
        
        # Configure the submitit Executor for this job
        executor = submitit.AutoExecutor(folder=shared_fs_path / "submitit_logs" / f"step_{i}")
        executor.update_parameters(
            job_name=f"{run_name}_step{i}",
            slurm_partition=launcher_cfg.get('partition'),
            cpus_per_task=launcher_cfg.cpus_per_task,
            mem_gb=launcher_cfg.mem_gb,
            gpus_per_task=launcher_cfg.get('gpus_per_task', 0),
            timeout_min=launcher_cfg.timeout_min
        )
        
        # Define I/O paths and worker arguments
        output_file = shared_fs_path / f"step_{i}_{algo_name}.pt"
        input_arg = f"data.precomputed_path={shared_fs_path / f'step_{i-1}_*.pt'}" if i > 0 else f"data={cfg.data._target_.split('.')[-1].lower()}"
        
        worker_overrides = [
            f"algorithm={algo_path}",
            input_arg,
            f"callbacks.embedding.save_embeddings.save_path={output_file}"
        ]

        print(f"\nSubmitting Step {i+1} ({algo_name})...")
        func = submitit.helpers.CommandFunction([
            "python", "-m", "manylatents.main", 
            f"--config-dir={Path.cwd() / 'manylatents/configs'}",
            *worker_overrides
        ])
        
        job = None
        with executor.batch():
            if previous_job:
                job = executor.submit(func, after=previous_job) # Chain dependency
            else:
                job = executor.submit(func)
        
        previous_job = job
        print(f"  --> Submitted as Job ID: {job.job_id}")

if __name__ == "__main__":
    main()