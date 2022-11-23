import argparse
import uvicorn
from logging import getLogger

from online_inference.testing_router import testing_router
from online_inference.app import app


logger = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testmode",
        action="store_true",
        help="Launch server in test mode, some new request may appear.",
    )

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

    if args.testmode:
        logger.warning("App launched in testing mode.")
        app.include_router(testing_router)

    uvicorn.run(
        app,  # type: ignore
        host=args.host,
        port=int(args.port),
        log_level="info",
    )


if __name__ == "__main__":
    main()
