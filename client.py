import argparse
import json
import sys
 
import requests
 
DEFAULT_URL = "http://localhost:8000"  # replace with your AWS endpoint, e.g. ALB DNS or EC2 public IP
 
 
def load_payload(data_str: str | None, file_path: str | None) -> dict:
    if file_path:
        with open(file_path) as f:
            return json.load(f)
    if data_str:
        return json.loads(data_str)
    print("Error: provide either --data or --file", file=sys.stderr)
    sys.exit(1)
 
 
def call_api(url: str, endpoint: str, payload: dict, timeout: int = 10) -> dict:
    full_url = f"{url.rstrip('/')}{endpoint}"
    try:
        response = requests.post(full_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"API returned an error: {e}", file=sys.stderr)
        print(f"Response body: {response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Failed to reach API at {full_url}: {e}", file=sys.stderr)
        sys.exit(1)
 
 
def main():
    parser = argparse.ArgumentParser(description="CLI client for the model inference API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the hosted model API")
 
    subparsers = parser.add_subparsers(dest="command", required=True)
 
    predict_parser = subparsers.add_parser("predict", help="Send a single record for prediction")
    predict_parser.add_argument("--data", help="Inline JSON string of a single record")
    predict_parser.add_argument("--file", help="Path to a JSON file containing a single record")
 
    batch_parser = subparsers.add_parser("predict-batch", help="Send multiple records for prediction")
    batch_parser.add_argument("--file", required=True, help="Path to a JSON file with {\"records\": [...]}")
 
    health_parser = subparsers.add_parser("health", help="Check API health")
 
    args = parser.parse_args()
 
    if args.command == "health":
        result = call_api(args.url, "/health", payload={})
        print(json.dumps(result, indent=2))
        return
 
    if args.command == "predict":
        payload = load_payload(args.data, args.file)
        result = call_api(args.url, "/predict", payload)
        print(json.dumps(result, indent=2))
        return
 
    if args.command == "predict-batch":
        with open(args.file) as f:
            payload = json.load(f)
        result = call_api(args.url, "/predict/batch", payload)
        print(json.dumps(result, indent=2))
        return
 
 
if __name__ == "__main__":
    main()
 