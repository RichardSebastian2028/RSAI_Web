from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    # Return HTML as a string
    return """
    <html>
        <head><title>My Page</title></head>
        <body>
            <h1>Hello from Python!</h1>
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)