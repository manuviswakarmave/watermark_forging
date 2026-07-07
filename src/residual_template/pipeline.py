

from pathlib import Path
import sys
import gc
from typing import List, Optional, Dict


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    EXPERIMENTS,
    GROUPS,
    TARGET_MAPPING,
    PRINT_PROGRESS,
    VERIFY_OUTPUTS,
    SAVE_TEMPLATE_VISUALS,
    get_experiment_by_name,
    get_experiment_output_dir,
)

from residual_template.template_extraction import (
    extract_all_templates,
    save_all_templates_numpy,
    save_all_template_visuals,
)

from residual_template.forgery import forge_experiment

from residual_template.visualization import save_comparison_visuals

from residual_template.image_ops import verify_png_folder


def clear_memory() -> None:
    gc.collect()


def print_experiment_summary(experiment: dict) -> None:
    print()
    print("=" * 80)
    print(f"Experiment: {experiment['name']}")
    print("=" * 80)

    for key, value in experiment.items():
        print(f"  {key}: {value}")

    print("=" * 80)


def run_one_experiment(experiment: dict) -> Dict:
    """
    Run one residual-template experiment.
    """

    print_experiment_summary(experiment)

    template_results = extract_all_templates(experiment)

    save_all_templates_numpy(
        experiment_name=experiment["name"],
        template_results=template_results,
    )

    if SAVE_TEMPLATE_VISUALS:
        save_all_template_visuals(
            experiment_name=experiment["name"],
            template_results=template_results,
        )

    forge_results = forge_experiment(
        experiment=experiment,
        template_results=template_results,
    )

    save_comparison_visuals(
        experiment_name=experiment["name"],
        forge_results=forge_results,
    )

    verification = None

    if VERIFY_OUTPUTS:
        output_dir = get_experiment_output_dir(experiment["name"])

        verification = verify_png_folder(
            folder=output_dir,
            groups=GROUPS,
            target_mapping=TARGET_MAPPING,
        )

        print()
        print(f"Verification for {experiment['name']}:")
        print(f"  Folder: {output_dir}")
        print(f"  Count:  {verification['count']}")
        print(f"  OK:     {verification['ok']}")

        if not verification["ok"]:
            print(f"  Missing: {verification['missing'][:20]}")
            print(f"  Extra:   {verification['extra'][:20]}")
            raise RuntimeError(f"Verification failed for {experiment['name']}")

    clear_memory()

    return {
        "experiment": experiment,
        "template_results": template_results,
        "forge_results": forge_results,
        "verification": verification,
    }


def run_residual_pipeline(
    experiment_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Run selected experiments.

    If experiment_names is None, run all configured experiments.
    """

    if experiment_names is None:
        experiments = EXPERIMENTS
    else:
        experiments = [
            get_experiment_by_name(name)
            for name in experiment_names
        ]

    print()
    print("=" * 80)
    print("Residual-template watermark forgery pipeline")
    print("=" * 80)
    print(f"Experiments to run: {[e['name'] for e in experiments]}")
    print("=" * 80)

    all_results = {}

    for experiment in experiments:
        result = run_one_experiment(experiment)
        all_results[experiment["name"]] = result

    print()
    print("=" * 80)
    print("Finished residual-template pipeline.")
    print("=" * 80)

    print()
    print("Next step:")
    print("  python src/residual_template/create_zip_residual.py --experiment <experiment_name>")
    print()
    print("Example:")
    print("  python src/residual_template/create_zip_residual.py --experiment gaussian_pos_a010")
    print("=" * 80)

    return all_results