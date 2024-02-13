from llm_service.admin_client import AdminClient
import argparse


parser = argparse.ArgumentParser(description="LLM Serving Admin Client")
parser.add_argument("--key", help="Admin Key", type=str, default="IloveRocknRoll")
parser.add_argument(
    "--mode",
    help="0: get model route; 1: delete model; 2: list models; 3: dump config; 4: reset unsupported models; 5: reset LLM service",
    type=int,
    required=True,
)
parser.add_argument(
    "-e",
    "--endpoint",
    help="LLM-Serving Endpoint",
    type=str,
    default="https://api.ncsa.ai/llm/admin",
)
parser.add_argument(
    "--model-name", help="Model Name", type=str, default="meta-llama/Llama-2-7b-chat-hf"
)
parser.add_argument("--model-type", help="Model Type", type=str, default="vllm_raw")
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--config-dump-path", type=str, default="latest_config.yaml")
args = parser.parse_args()

admin_client = AdminClient(args.endpoint, args.key)


if args.mode == 0:
    print(f"Requesting route for model {args.model_name}")
    res = admin_client.get_model_route(args.model_name, args.model_type, args.num_gpus)
    print(res)

elif args.mode == 1:
    print(f"Deleting model {args.model_name}")
    res = admin_client.delete_model(args.model_name)
    print(res)

elif args.mode == 2:
    print("Listing all models")
    res = admin_client.list_models(print_models=True)

elif args.mode == 3:
    print(f"Dumping config to file {args.config_dump_path}")
    admin_client.dump_config(args.config_dump_path)

elif args.mode == 4:
    print("Resetting unsupported models")
    res = admin_client.reset_unsupported()
    print(res)

elif args.mode == 5:
    print("Resetting LLM service")
    res = admin_client.reset_llm_service()
    print(res)

else:
    print("Invalid mode")
