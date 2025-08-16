import hydra
from hydra import initialize, compose
import sys
from pathlib import Path
import uuid
import submitit
import manylatents.configs  # Trigger ConfigStore registration

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
            algo_params = {}
        else:
            # Handle dictionary format like {latent/pca: {param: value}}
            algo_path = list(algo_name.keys())[0] if isinstance(algo_name, dict) else str(algo_name)
            algo_params = algo_name[algo_path] if isinstance(algo_name, dict) else {}
        
        # Load full config with hydra defaults to get launcher configuration  
        try:
            algo_name_only = algo_path.split('/')[-1] if algo_path else "pca"
            override_path = f"algorithms/latent={algo_name_only}" if "latent" in algo_path else f"algorithms/lightning={algo_name_only}"
            
            with initialize(config_path="manylatents/configs", version_base=None):
                step_cfg = compose(config_name="config", overrides=[
                    override_path,
                    "hydra=default"  # Ensure hydra config is loaded
                ])
        except Exception as e:
            print(f"Warning: Could not load config for {algo_path}, using defaults: {e}")
            step_cfg = None
        
        launcher_cfg = step_cfg.hydra.launcher if step_cfg and hasattr(step_cfg, 'hydra') and hasattr(step_cfg.hydra, 'launcher') else {
            'cpus_per_task': 4,
            'mem_gb': 8, 
            'timeout_min': 60,
            'partition': 'main'
        }
        
        # Configure the submitit Executor for this job
        executor = submitit.AutoExecutor(folder=shared_fs_path / "submitit_logs" / f"step_{i}")
        # Handle both dict and DictConfig access patterns
        def get_launcher_param(cfg, key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            else:
                return getattr(cfg, key, default)
        
        executor.update_parameters(
            job_name=f"{run_name}_step{i}",
            slurm_partition=get_launcher_param(launcher_cfg, 'partition', 'main'),
            cpus_per_task=get_launcher_param(launcher_cfg, 'cpus_per_task', 4),
            mem_gb=get_launcher_param(launcher_cfg, 'mem_gb', 8),
            gpus_per_task=get_launcher_param(launcher_cfg, 'gpus_per_task', 0),
            timeout_min=get_launcher_param(launcher_cfg, 'timeout_min', 60)
        )
        
        # Define I/O paths and worker arguments
        output_file = shared_fs_path / f"step_{i}_{algo_name}.pt"
        # Use swissroll as default data (from the pipeline config)
        input_arg = f"data.precomputed_path={shared_fs_path / f'step_{i-1}_*.pt'}" if i > 0 else "data=swissroll"
        
        worker_overrides = [
            f"algorithms/latent={algo_path.split('/')[-1]}" if "latent" in algo_path else f"algorithms/lightning={algo_path.split('/')[-1]}",
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