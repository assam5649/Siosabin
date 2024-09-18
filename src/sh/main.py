from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/post-json', methods=['POST'])
def post_json():
    print(request.data)  # Print raw request data for debugging
    try:
        data = request.get_json()  # Attempt to parse the JSON
        if data is None:
            return "Bad Request: No JSON received", 400
        return jsonify(data)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400

if __name__ == "__main__":
    app.run(debug=True)
