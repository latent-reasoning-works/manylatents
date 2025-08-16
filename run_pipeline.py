import hydra
from hydra import initialize, compose
import sys
from pathlib import Path
import uuid
import submitit
import manylatents.configs  # Trigger ConfigStore registration

def main():
    # Agent/user calls this script like: `python run_pipeline_new.py +experiment/pipeline=multiple_algorithms`
    overrides = sys.argv[1:]
    with initialize(config_path="manylatents/configs", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    pipeline_steps = cfg.pipeline
    
    # Get the save format from a pipeline step config that includes callbacks
    # Load the first step's experiment config to get the save format
    first_step = pipeline_steps[0]
    first_experiment = first_step.get('experiment')
    with initialize(config_path="manylatents/configs", version_base=None):
        step_cfg = compose(config_name="config", overrides=[f"experiment={first_experiment}"])
    save_format = step_cfg.callbacks.embedding.save_embeddings.save_format
    
    run_name = cfg.get('name', 'pipeline_run')
    shared_fs_path = Path(f"/network/scratch/c/cesar.valdez/manyLatents/pipeline_runs/{run_name}-{uuid.uuid4().hex[:8]}")
    shared_fs_path.mkdir(parents=True, exist_ok=True)
    previous_job = None
    previous_output_file = None  # Track exact output file for next step

    print(f"Orchestrating pipeline '{run_name}' with steps: {pipeline_steps}")
    print(f"Using save format: {save_format}")

    for i, step_config in enumerate(pipeline_steps):
        # Extract experiment and overrides from pipeline step
        experiment_name = step_config.get('experiment')
        step_overrides = step_config.get('overrides', [])
        
        # Get experiment name for job naming
        step_name = experiment_name.replace('pipeline_step_', '')
        
        # Load experiment config to get launcher configuration  
        try:
            with initialize(config_path="manylatents/configs", version_base=None):
                step_cfg = compose(config_name="config", overrides=[
                    f"experiment={experiment_name}",
                    "hydra=default"  # Ensure hydra config is loaded
                ])
        except Exception as e:
            print(f"Warning: Could not load experiment {experiment_name}, using defaults: {e}")
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
            slurm_job_name=f"{run_name}_step{i}",
            slurm_partition=get_launcher_param(launcher_cfg, 'partition', 'main'),
            cpus_per_task=get_launcher_param(launcher_cfg, 'cpus_per_task', 4),
            mem_gb=get_launcher_param(launcher_cfg, 'mem_gb', 8),
            slurm_gpus_per_task=get_launcher_param(launcher_cfg, 'gpus_per_task', 0),
            timeout_min=get_launcher_param(launcher_cfg, 'timeout_min', 60)
        )
        
        # Define I/O paths using the format from config
        output_file = shared_fs_path / f"step_{i}_{step_name}.{save_format}"
        
        # Build worker overrides starting with experiment
        worker_overrides = [f"experiment={experiment_name}"]
        
        # Add data input configuration
        if i > 0:
            # For steps after the first, use precomputed data from previous step
            worker_overrides.append(f"data.precomputed_path={previous_output_file}")
        else:
            # For first step, ensure data source is specified in pipeline config
            # The data override should already be in step_overrides from pipeline config
            pass
        
        # Add custom save path for pipeline chaining
        worker_overrides.append(f"callbacks.embedding.save_embeddings.save_path={output_file}")
        
        # Add step-specific overrides
        worker_overrides.extend(step_overrides)

        print(f"\nSubmitting Step {i+1} ({step_name})...")
        
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
        previous_output_file = output_file  # Store exact output path for next step
        print(f"  --> Submitted as Job ID: {job.job_id}")

if __name__ == "__main__":
    main()