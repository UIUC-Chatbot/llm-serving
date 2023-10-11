from flask import Flask, request, jsonify

app = Flask(__name__)

# @app.route('/api/codeblocks', methods=['POST'])


def receive_code(setup_code, predict_code):
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

    # Execute the setup code
    exec(setup_code)

    # Execute the predict code
    predict_result = exec(predict_code)

    return jsonify({"status": "success", "message": "Code blocks received, printed, and executed.", "predict_result": str(predict_result)})

  except Exception as e:
    return jsonify({"status": "error", "message": str(e)})


class Predictor():
  def __init__(self, setup_code, predict_code): 
    self.model = exec(setup_code)
    self.predict = predict_code

  def run(self, input):
    return self.predict(input)
  


if __name__ == '__main__':


  app.run(debug=True)

  setup_code = """
  return pipeline(model='HuggingFaceH4/zephyr-7b-alpha', device_map='auto')
  """  

  # TODO: figure out how to pass a parameter into the user-defined predict_code function
  predict_code = """
  return pipe(prompt, **pipeline_kwargs)
  """
  input = "How many helicopters can a human eat in one sitting?"

  receive_code()