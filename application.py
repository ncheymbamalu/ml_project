from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_record():
    if request.method == "GET":
        return render_template("home.html")
    else:
        record = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score"))
        )

        df_record = record.convert_to_dataframe()
        prediction = min(PredictPipeline().predict(df_record), 100)
        return render_template("home.html", results=int(prediction))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
