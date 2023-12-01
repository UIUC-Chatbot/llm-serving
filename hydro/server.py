from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

app = Flask(__name__)
llm = LLM(model="mistralai/Mistral-7B-v0.1")

@app.route('/infer', methods=['POST'])
def infer():
    
    data = request.get_json()
    print(f"Got request with data: {data}")
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95)
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    print("Starting up main")
    app.run(host='0.0.0.0', port=5000)