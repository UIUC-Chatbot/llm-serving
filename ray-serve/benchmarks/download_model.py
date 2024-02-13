import argparse
import helpers
import subprocess

parser = argparse.ArgumentParser(
    description="Test downloading models from Hugging Face"
)
parser.add_argument("--endpoint", "-e", default="http://localhost:5004/llm/admin")
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


res = helpers.reset_llm_service(args.endpoint)
print(res)
