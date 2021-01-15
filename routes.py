import forms
import inference
from flask import render_template
from app import app


@app.route("/")
@app.route("/index", methods=["GET", "POST"])
def home():
    form = forms.AddTaskForm()
    if form.validate_on_submit():
        print("Submitted", form.title.data)
        text = inference.sample(prime=form.title.data)
        text_new = text.replace("\n", "<br>")
        return render_template("index.html", form=form,
                               title=form.title.data,
                               data=text_new)
    return render_template("index.html", form=form)

