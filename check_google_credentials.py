"""
GenAI diagnostic / model lister.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence

import coloredlogs
import google.auth
from google import genai
from google.auth import compute_engine
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2 import credentials as oauth2_creds
from google.oauth2 import service_account

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)


def dump_env_vars(*names: str) -> None:
    """Log selected environment variables and helpful hints."""
    if not names:
        names = (
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_API_KEY",
        )

    logger.info("Environment variables:")
    for name in names:
        value = os.getenv(name)
        logger.info("  %-35s = %r", name, value)


def describe_credentials(creds: google.auth.credentials.Credentials) -> None:
    """Emit a human-friendly description of credential type in use."""
    if isinstance(creds, service_account.Credentials):
        msg = "service-account key file"

    elif isinstance(creds, oauth2_creds.Credentials):
        msg = "gcloud user (OAuth) credentials"
    elif isinstance(creds, compute_engine.Credentials):
        msg = "Compute Engine / metadata-server credentials"
    else:
        msg = f"unknown credentials ({type(creds).__name__})"

    logger.info("Using %s", msg)


def list_models(project: str, location: str, vertexai: bool) -> None:
    """Print available models for the given project/location.

    Handles both old and new versions of the SDK where the attribute name for
    generation methods may differ (e.g. `supported_generation_methods` vs.
    `generation_methods`).
    """
    client = genai.Client(project=project, location=location, vertexai=vertexai)

    logger.info("Available models:")
    for model in client.models.list():
        methods = (
            getattr(model, "supported_generation_methods", None) or getattr(model, "generation_methods", None) or "?"
        )
        logger.info("  • %s (%s) – methods=%s", model.display_name, model.name, methods)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags with environment-aware defaults."""
    parser = argparse.ArgumentParser(description="Quick Vertex AI GenAI diagnostic utility.")

    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project (default: $GOOGLE_CLOUD_PROJECT or ADC project).",
    )

    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "us-west2"),
        help="Vertex AI location/region (default: us-west2 or $GOOGLE_CLOUD_LOCATION).",
    )

    # Determine default for vertexai from env-var (default True)
    default_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in {"1", "true", "yes"}

    parser.add_argument(
        "--vertexai",
        action="store_true",
        default=default_vertexai,
        help=(
            "Use Vertex AI backend instead of direct PaLM API. "
            "Defaults to the value of $GOOGLE_GENAI_USE_VERTEXAI (True if unset)."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING…).",
    )

    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="If set, include full tracebacks in error logs.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level.upper())

    creds, adc_project = google.auth.default()
    describe_credentials(creds)

    project = args.project or adc_project
    if not project:
        logger.error("No project could be determined. Provide --project or set GOOGLE_CLOUD_PROJECT.")
        # Continue execution but skip model listing
        return 0

    dump_env_vars()

    try:
        creds.refresh(Request())
    except RefreshError as exc:
        if args.verbose_errors:
            logger.exception("Failed to refresh credentials: %s", exc)
        else:
            logger.error("Failed to refresh credentials: %s", exc)
        # Proceed without refreshed credentials; downstream calls may still fail.
        pass

    try:
        list_models(project=project, location=args.location, vertexai=args.vertexai)
    except Exception as exc:  # noqa: BLE001
        if args.verbose_errors:
            logger.exception("Unexpected error while listing models: %s", exc)
        else:
            logger.error("Unexpected error while listing models: %s", exc)
        # Do not fail the entire script
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
