from flask import Flask,request,render_template

from src.pipeline.prediction_pipeline import PredictPipeline,CustomData

app=Flask(__name__,static_url_path='/static')


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("house.html")
    else:
        data=CustomData(
            area=float(request.form.get("area")),
            bedroom=float(request.form.get("bedroom")),
            bathroom=float(request.form.get("bathroom")),
            seller_type=request.form.get("seller"),
            layout_type=request.form.get("layout"),
            property_type=request.form.get("property"),
            furnish_type=request.form.get("furnish"),
            locality=request.form.get("locality"),
            city=request.form.get("city")
        )


        final_data=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()

        pred=predict_pipeline.predict(final_data)

        result=round(pred[0],2)
        result = f"₹{int(result)-1000} - ₹{int(result)+1000}"

        return render_template("result.html",final_result=result)
    
@app.route('/load',methods=["GET","POST"])
def load_page():
    if request.method == "POST" or "GET":
        return render_template("loader.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)