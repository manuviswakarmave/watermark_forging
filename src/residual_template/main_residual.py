
from pathlib import Path
import sys
import argparse


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.pipeline import run_residual_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run residual-template watermark forgery experiments."
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Experiment names to run. "
            "Example: --experiments gaussian_pos_a010 bilateral_pos_a010"
        ),
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only gaussian_pos_a010 and bilateral_pos_a010.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        experiment_names = [
            "gaussian_pos_a010",
            "bilateral_pos_a010",
        ]
    else:
        experiment_names = args.experiments

    run_residual_pipeline(
        experiment_names=experiment_names,
    )


if __name__ == "__main__":
    main()