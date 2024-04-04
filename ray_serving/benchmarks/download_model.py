import argparse
from client.admin_client import AdminClient
import subprocess
import time


parser = argparse.ArgumentParser(
    description="Test downloading models from Hugging Face"
)
parser.add_argument("-e", "--endpoint", default="http://localhost:5004/llm/admin")
parser.add_argument("-k", "--key", help="Admin Key", type=str, default="IloveRocknRoll")
parser.add_argument(
    "--model-name",
    help="Model Name",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
)
args = parser.parse_args()

try:
    result = subprocess.run(
        "rm /home/qinrenz2/.cache/huggingface/hub/* -r",
        capture_output=True,
        text=True,
        shell=True,
    )
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
print(result.stdout)

admin_client = AdminClient(args.endpoint, args.key)
res = admin_client.reset_llm_service()
print(res)

start_time = time.time()
res = admin_client.get_model_route(args.model_name, "vllm_openai", 1)
end_time = time.time()
print(f"Time to download model {args.model_name}: {end_time - start_time} seconds.")
