# VLLM Flask Server on Hydro

This repository contains a Flask server that uses the VLLM library to run a language model for text generation. The model used in this example is `mistralai/Mistral-7B-v0.1`.

## Files

- `server.py`: This is the main Flask server script. It sets up an API endpoint at `/infer` that accepts POST requests with a JSON body containing a prompt. The server uses the VLLM library to generate text based on the prompt and returns it in the response.

- `requirements.txt`: This file lists the Python dependencies that need to be installed for the server to run.

- `startup.sh`: This is a SLURM job script that sets up the environment and starts the Flask server. It checks if a Python virtual environment exists, creates it if it doesn't, installs the dependencies, and runs the server.

## Usage

1. **Clone this repository and navigate to its directory.**

2. **Ensure necessary modules are loaded**: The versions used in this example are CUDA 12.2.1 and Python 3.9.13.

3. **Submit the `startup.sh` script to SLURM**:

```bash
   sbatch --account=<account-name> startup.sh
```


4. **Start the Flask server**: Once the job starts, the Flask server will be running and listening for requests on port 5000.

5. **Send a POST request** to the `/infer` endpoint with a JSON body containing a prompt. For example:


```
curl -X POST -H "Content-Type: application/json" -d '{"prompt":"Once upon a time"}' http://localhost:5000/infer
```


6. **Receive the response**: The server will respond with a JSON object containing the generated text.

## Note

This server is set up for demonstration purposes and is not intended for production use. For a production environment.
