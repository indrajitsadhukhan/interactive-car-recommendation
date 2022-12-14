from flask import Flask, render_template,request,jsonify
from chat import bulk_response
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

# @app.route("/",methods=["GET"])
@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    #TODO: Check if text is valid
    responses=bulk_response(text)
    message={"answer":responses}
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)
