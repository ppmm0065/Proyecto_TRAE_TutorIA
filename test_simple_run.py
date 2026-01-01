from flask import Flask
import sys

print("Starting simple test...")
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World"

if __name__ == '__main__':
    print("Running app on 5001...")
    try:
        app.run(port=5001)
    except Exception as e:
        print(f"Error: {e}")
