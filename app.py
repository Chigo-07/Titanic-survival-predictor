from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model.h5", "rb") as f:
    model, scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        data = [float(x) for x in request.form.values()]
        data = scaler.transform([data])
        result = model.predict(data)[0]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
