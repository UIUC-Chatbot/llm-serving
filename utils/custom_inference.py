import contextlib
import io
from sys import stderr

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/custom_inference', methods=['POST'])
def receive_code():
  try:
    # setup_code, predict_code, input
    print("Top of custom_inference")
    data = request.json
    setup_code = data.get('setup', '')
    predict_code = data.get('predict', '')
    input = data.get('input', '')

    print("Setup Code Block:")
    print("-----------------")
    print(setup_code)
    print()

    print("Predict Code Block:")
    print("-------------------")
    print(predict_code)
    
    print("Input:")
    print("-------------------")
    print(input)

    # Setup 
    setup_globals = {}
    setup_locals = {}
    stderr = io.StringIO()
    try:
      with contextlib.redirect_stderr(stderr):
        exec(setup_code, setup_globals, setup_locals)
    except Exception as e:
      print(f"Error in setup: {e}")

    pipe = setup_locals.get('pipe')

    # Print globals and locals
    print("Globals:", setup_globals)
    print("Locals:", setup_locals)
    print("STDERR:", stderr.getvalue())

    # Predict
    predict_globals = {}
    predict_locals = {'pipe': pipe, 'input': input}
    stderr = io.StringIO()

    try:
      with contextlib.redirect_stderr(stderr):
        exec(predict_code, predict_globals, predict_locals)
    except Exception as e:
      print(f"Error in predict: {e}")

    print(f"RESULT: {predict_locals.get('output')}")
    print("STDERR:", stderr.getvalue())

    return {"status": "success", "message": "Setup runtime, predict runtime", "predict_result": str(predict_locals.get('output'))}

  except Exception as e:
    return {"status": "error", "message": str(e)}  


if __name__ == '__main__':
  app.run(debug=True)