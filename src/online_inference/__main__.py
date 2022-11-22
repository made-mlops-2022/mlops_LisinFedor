import argparse
import uvicorn

from online_inference.app import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        required=False,
        default="0.0.0.0",
        help="Specify server host. Default 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        required=False,
        default=80,
        help="Specify port. Default 80",
    )

    args = parser.parse_args()

    uvicorn.run(
        app,  # type: ignore
        host=args.host,
        port=int(args.port),
        log_level="info",
    )


if __name__ == "__main__":
    main()