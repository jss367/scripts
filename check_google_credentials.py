import logging
import os

import google
import google.auth
import google.oauth2.service_account
from google import genai
from google.auth import compute_engine
from google.oauth2 import credentials as oauth2_creds
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
os.environ["GOOGLE_CLOUD_LOG_LEVEL"] = "debug"

print(f"genai.__version__: {genai.__version__}")

# This finds whatever ADC would use
creds, project = google.auth.default()
print("\nApplication Default Credentials (ADC) Info:")
print("ADC project:", project)
print("Cred type  :", creds.__class__.__name__)

client = genai.Client(vertexai=True, project="visotrustdev", location="us-west2")

# Inspect the type
print("Credential type:", type(creds))

# Detailed credential type information
if isinstance(creds, service_account.Credentials):
    print(
        "→ Using a service account key file. This is common for applications running outside Google Cloud or when explicit service account impersonation is used."
    )
elif isinstance(creds, oauth2_creds.Credentials):
    print(
        "→ Using user (gcloud) credentials. This is typical for local development when you've authenticated via `gcloud auth application-default login`."
    )
elif isinstance(creds, compute_engine.Credentials):
    print(
        "→ Using Compute Engine / metadata-server credentials. This is standard for applications running on Google Cloud services like GCE, GKE, Cloud Functions, etc."
    )
else:
    print(f"→ Some other credential type: {creds}. This might be a custom credential implementation.")


print("\nListing available models with details:")
for model in client.models.list():
    print(f"  Model Display Name: {model.display_name}")
    print(f"  Model Resource Name: {model.name}")
    print(f"  Supported Generation Methods: {model.supported_generation_methods}")
    print("-" * 20)


print(f"client.models.vertexai: {client.models.vertexai}")

# Or more precisely:
if isinstance(creds, service_account.Credentials):
    print("→ Using a service account key file")
elif isinstance(creds, oauth2_creds.Credentials):
    print("→ Using user (gcloud) credentials")
elif isinstance(creds, compute_engine.Credentials):
    print("→ Using Compute Engine / metadata-server credentials")
else:
    print("→ Some other credential type:", creds)


request = google.auth.transport.requests.Request()
creds.refresh(request)

token_parts = creds.token.split(".")
print(f"token_parts: {token_parts}")


def dump_vars(*names):
    print("\nEnvironment Variable Check:")
    for name in names:
        value = os.getenv(name)
        comment = ""
        if name == "GOOGLE_GENAI_USE_VERTEXAI":
            comment = " (Should be 'true' or 'True' to use Vertex AI backend for GenAI client. If not set or 'false', it uses the PaLM API directly.)"
        elif name == "GOOGLE_CLOUD_PROJECT":
            comment = " (Specifies the Google Cloud project. If not set, the client library might infer it from ADC or other sources.)"
        elif name == "GOOGLE_CLOUD_LOCATION":
            comment = " (Specifies the default location/region for Vertex AI services.)"
        elif name == "GOOGLE_APPLICATION_CREDENTIALS":
            comment = " (Path to a service account key JSON file. If set, ADC will use this file.)"
        elif name == "GOOGLE_API_KEY":
            comment = " (API key for PaLM API. Should generally be None or not set when using Vertex AI, as ADC is preferred for authentication.)"
        print(f"{name} = {value!r}{comment}")


dump_vars(
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_API_KEY",
)
