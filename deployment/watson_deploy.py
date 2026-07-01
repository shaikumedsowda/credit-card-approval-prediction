"""
watson_deploy.py
------------------
Deploys the trained credit-card-approval model (models/best_model.pkl)
to IBM Watson Machine Learning, so it can be hosted on the cloud for
scalable, real-time predictions accessed through a REST endpoint.

USAGE
-----
1. pip install ibm-watson-machine-learning
2. Set the environment variables below (or edit them directly):
     WML_API_KEY       - your IBM Cloud API key
     WML_URL           - your Watson ML service region URL,
                          e.g. https://us-south.ml.cloud.ibm.com
     WML_SPACE_ID       - the deployment space ID to publish into
3. Run:  python deployment/watson_deploy.py

This script is intentionally self-contained and safe to run repeatedly -
it stores the deployment ID it creates in deployment/watson_deployment.json
so subsequent scoring calls know where to send requests.
"""

import os
import json
import pickle

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEPLOY_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watson_deployment.json")


def get_credentials():
    api_key = os.environ.get("WML_API_KEY")
    url = os.environ.get("WML_URL", "https://us-south.ml.cloud.ibm.com")
    space_id = os.environ.get("WML_SPACE_ID")

    missing = [n for n, v in [("WML_API_KEY", api_key), ("WML_SPACE_ID", space_id)] if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them before running this script - see the module docstring."
        )
    return api_key, url, space_id


def main():
    api_key, url, space_id = get_credentials()

    from ibm_watson_machine_learning import APIClient  # imported lazily so the
    # rest of the project doesn't require this dependency unless you actually deploy

    wml_credentials = {"apikey": api_key, "url": url}
    client = APIClient(wml_credentials)
    client.set.default_space(space_id)

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)

    print(f"Publishing model '{meta['best_model_name']}' to Watson Machine Learning...")

    sw_spec_uid = client.software_specifications.get_uid_by_name("runtime-23.1-py3.10")

    model_props = {
        client.repository.ModelMetaNames.NAME: "credit-card-approval-model",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_uid,
    }

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    published_model = client.repository.store_model(
        model=model_obj, meta_props=model_props
    )
    model_uid = client.repository.get_model_id(published_model)

    deployment_props = {
        client.deployments.ConfigurationMetaNames.NAME: "credit-card-approval-deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment = client.deployments.create(model_uid, meta_props=deployment_props)
    deployment_id = client.deployments.get_id(deployment)
    scoring_url = client.deployments.get_scoring_href(deployment)

    with open(DEPLOY_META_PATH, "w") as f:
        json.dump({"deployment_id": deployment_id, "scoring_url": scoring_url}, f, indent=2)

    print(f"Deployed. Deployment ID: {deployment_id}")
    print(f"Scoring endpoint: {scoring_url}")
    print(f"Saved deployment metadata to {DEPLOY_META_PATH}")


if __name__ == "__main__":
    main()
