from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

model = joblib.load("lightgbm_model.pkl")
model_columns = joblib.load("model_columns.pkl")

BEST_THRESHOLD = 0.4


@app.route("/")
def home():
    return jsonify({"message": "Bank Telemarketing Model API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether a bank client will subscribe to a term deposit
    ---
    tags:
      - Bank Telemarketing Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - age
            - job
            - marital
            - education
            - default
            - housing
            - loan
            - contact
            - month
            - day_of_week
            - campaign
            - pdays
            - previous
            - poutcome
            - emp.var.rate
            - cons.price.idx
            - cons.conf.idx
            - euribor3m
            - nr.employed
          properties:
            age:
              type: integer
              example: 35
            job:
              type: string
              example: admin.
            marital:
              type: string
              example: married
            education:
              type: string
              example: university.degree
            default:
              type: string
              example: no
            housing:
              type: string
              example: yes
            loan:
              type: string
              example: no
            contact:
              type: string
              example: cellular
            month:
              type: string
              example: may
            day_of_week:
              type: string
              example: mon
            campaign:
              type: integer
              example: 1
            pdays:
              type: integer
              example: 999
            previous:
              type: integer
              example: 0
            poutcome:
              type: string
              example: nonexistent
            emp.var.rate:
              type: number
              example: -1.8
            cons.price.idx:
              type: number
              example: 93.9
            cons.conf.idx:
              type: number
              example: -42.7
            euribor3m:
              type: number
              example: 1.1
            nr.employed:
              type: number
              example: 5099.1
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            prediction:
              type: integer
              example: 0
            probability:
              type: number
              example: 0.237
            threshold:
              type: number
              example: 0.4
    """
    data = request.get_json()

    input_df = pd.DataFrame([data])

    input_df["contact_intensity"] = input_df["campaign"] / (input_df["previous"] + 1)
    input_df["was_previously_contacted"] = (input_df["pdays"] != 999).astype(int)
    input_df["pdays"] = input_df["pdays"].replace(999, -1)

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prob = model.predict_proba(input_encoded)[0][1]
    prediction = 1 if prob >= BEST_THRESHOLD else 0

    return jsonify({
        "prediction": int(prediction),
        "probability": float(prob),
        "threshold": BEST_THRESHOLD
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)