
from pathlib import Path
import sys
import zipfile
import argparse


SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from residual_template.config_residual import (
    GROUPS,
    TARGET_MAPPING,
    SUBMISSION_EXPERIMENT_NAME,
    RESIDUAL_SUBMISSION_ZIP_PATH,
    RESIDUAL_EXPECTED_SUBMISSION_COUNT,
    get_experiment_output_dir,
)

from residual_template.image_ops import (
    sort_by_numeric_stem,
    verify_png_folder,
)


def create_submission_zip(
    experiment_name: str,
) -> Path:
    source_dir = get_experiment_output_dir(experiment_name)

    verification = verify_png_folder(
        folder=source_dir,
        groups=GROUPS,
        target_mapping=TARGET_MAPPING,
    )

    if not verification["ok"]:
        raise RuntimeError(
            f"Cannot zip invalid folder: {source_dir}\n"
            f"Count: {verification['count']}\n"
            f"Missing: {verification['missing'][:20]}\n"
            f"Extra: {verification['extra'][:20]}"
        )

    png_files = sort_by_numeric_stem(
        [path for path in source_dir.glob("*.png") if path.is_file()]
    )

    if len(png_files) != RESIDUAL_EXPECTED_SUBMISSION_COUNT:
        raise RuntimeError(
            f"Expected {RESIDUAL_EXPECTED_SUBMISSION_COUNT} images, "
            f"found {len(png_files)}."
        )

    output_zip = RESIDUAL_SUBMISSION_ZIP_PATH
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    if output_zip.exists():
        output_zip.unlink()

    print()
    print("=" * 80)
    print("Creating residual-template submission zip")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Source:     {source_dir}")
    print(f"Zip:        {output_zip}")
    print(f"Files:      {len(png_files)}")
    print("=" * 80)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for image_path in png_files:
            zipf.write(image_path, arcname=image_path.name)

    print()
    print("Created:")
    print(f"  {output_zip}")

    return output_zip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create submission.zip for residual-template outputs."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=SUBMISSION_EXPERIMENT_NAME,
        help=f"Experiment to zip. Default: {SUBMISSION_EXPERIMENT_NAME}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    create_submission_zip(
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()