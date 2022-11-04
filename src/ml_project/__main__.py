import argparse
from typing import Dict
from ml_project.train import train_pipeline
from ml_project.predict import load_and_predict


def parse_args():
    parser = argparse.ArgumentParser("MLOps HW1")
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument(
        "--train", action="store_true", help="run train pipline using config.yml file."
    )

    main_group.add_argument(
        "--predict",
        action="store_true",
        help="make prediction --input using --modelid or "
        "last (from config.yml) saved model; store to --outfile",
    )

    predict_group = parser.add_mutually_exclusive_group()

    predict_group.add_argument(
        "--modelid",
        help="id of model from mlflow storage",
    )

    predict_group.add_argument(
        "--modelpath",
        help="path to model; specify only one of the two arguments 'id' or 'path'",
    )

    parser.add_argument(
        "--outfile",
        help="file path to store prediction in 'my/path/file.csv' "
        "format; required argument for prediction",
    )

    parser.add_argument(
        "--input", help="path to file for prediction; required argument for prediction",
    )

    return parser.parse_args()


def create_prediction_kwargs(parsed_args) -> Dict[str, str]:
    if parsed_args.input is None:
        raise argparse.ArgumentError(
            parsed_args.input, "--input is required for prediction.",
        )
    if parsed_args.outfile is None:
        raise argparse.ArgumentError(
            parsed_args.outfile, "--outfile is required for prediction.",
        )

    kwargs = {}

    if parsed_args.modelid:
        kwargs["model_id"] = parsed_args.modelid

    if parsed_args.modelpath:
        kwargs["model_path"] = parsed_args.modelpath

    if parsed_args.input:
        kwargs["data_path"] = parsed_args.input
    else:
        raise argparse.ArgumentError(
            parsed_args.input, "--input path is required for prediction",
        )

    if parsed_args.outfile:
        kwargs["out_csv_file_path"] = parsed_args.outfile
    else:
        raise argparse.ArgumentError(
            parsed_args.outfile, "--outfile path is required for prediction",
        )

    return kwargs


def main():
    parsed_args = parse_args()
    if parsed_args.predict:
        kwargs = create_prediction_kwargs(parsed_args)
        load_and_predict(**kwargs)
    else:
        train_pipeline()


if __name__ == "__main__":
    main()
