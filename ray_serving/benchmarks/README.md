# Benchmarks

To run the benchmarks, first navigate to the **/ray-serve/** directory. Then, run the following command:

```bash
python -m benchmarks.$BENCHMEARK_NAME
```

For example,

```bash
python -m benchmarks.download_model
```

- [x] Downloading and loading models from Huggingface: meta-llama/Llama-2-7b-chat-hf, 103.6 seconds.
- [x] Time to First Token: meta-llama/Llama-2-7b-chat-hf, 0.039 seconds.
- [x] Time per Output Token: meta-llama/Llama-2-7b-chat-hf, 0.038 seconds.
- [x] Time to switch from one model to another: to meta-llama/Llama-2-7b-chat-hf, 33.9 seconds.
