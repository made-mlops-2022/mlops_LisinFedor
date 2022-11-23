"""Scripts for prediction request to fastapi app."""
import argparse
import sys
import requests
import csv
import json
from pathlib import Path
from typing import Union, List, Dict, Any

filepath = Path(__file__).parent


def get_patients(path: Union[Path, str]) -> List[List[str]]:
    """Load csv data."""
    pat_data = []
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            pat_data.append(row)
    return pat_data


def send_post_matrix(
    data_pat: List[List[str]], host: str, port: str
) -> List[Dict[str, Any]]:
    """Send post request to /predict/many/matrix."""
    url = f"http://{host}:{port}/predict/many/matrix"
    req = {"data_mat": data_pat}
    resp = requests.post(url, json=req)
    return resp.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host", default="localhost", action="store", help="host address"
    )
    parser.add_argument("--port", default="80", action="store", help="port")

    parser.add_argument(
        "--input",
        required=True,
        action="store",
        help="path to input data in csv format",
    )
    parser.add_argument(
        "--output",
        required=False,
        action="store",
        help="path to save file; default print to stdout",
    )

    args = parser.parse_args()
    inpath = args.input
    host = args.host
    port = args.port

    df = get_patients(inpath)
    resp = send_post_matrix(df, host, port)

    json_resp = {"items": resp}

    if args.output:
        with open(args.output, "w") as json_file:
            json.dump(json_resp, json_file, indent=4)
    else:
        str_js = json.dumps(json_resp, indent=4)
        sys.stdout.write(str_js)
