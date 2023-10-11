import contextlib
import io
from sys import stderr


def receive_code(setup_code, predict_code, input):
  try:
    # data = request.json
    # setup_code = data.get('setup', '')
    # predict_code = data.get('predict', '')

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
      print(f"Error: {e}")

    exec(setup_code, setup_globals, setup_locals)
    pipe = setup_locals.get('pipe')
    print(f"done setting up the pipeline: {pipe}")

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
      print(f"Error: {e}")

    print(f"RESULT: {predict_locals.get('output')}")
    print("STDERR:", stderr.getvalue())

    return {"status": "success", "message": "Code blocks received, printed, and executed.", "predict_result": str(predict_result)}

  except Exception as e:
    return {"status": "error", "message": str(e)}


# class Predictor():
#   def __init__(self, setup_code, predict_code): 
#     self.model = exec(setup_code)
#     self.predict = predict_code

#   def run(self, input):
#     return self.predict(input)
  


if __name__ == '__main__':
  setup_code = """
from transformers import pipeline
print("Starting up the pipeline")
pipe = pipeline(model='HuggingFaceH4/zephyr-7b-alpha', device_map='auto')
"""

  # TODO: figure out how to pass a parameter into the user-defined predict_code function
  predict_code = """output = pipe(input)"""
  input = "How many helicopters can a human eat in one sitting?"

  receive_code(setup_code, predict_code, input)