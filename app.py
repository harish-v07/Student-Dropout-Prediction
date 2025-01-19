from flask import Flask, request, render_template
import pickle



app = Flask(__name__)


model =pickle.load(open("dropout_model.pkl", "rb"))
@app.route("/")
def Home():
    return render_template("dropout.html")

@app.route("/predict", methods=["POST"])
def predict():
    marital_status = request.form.get('marital_status')
    application_mode = request.form.get('application_mode')
    application_order = request.form.get('application_order')
    course = request.form.get('course')
    attendance = request.form.get('attendance')
    prev_qualification = request.form.get('prev_qualification')
    prev_qualification_grade = request.form.get('prev_qualification_grade')
    nationality = request.form.get('nationality')
    mother_qualification = request.form.get('mother_qualification')
    father_qualification = request.form.get('father_qualification')
    mother_occupation = request.form.get('mother_occupation')
    father_occupation = request.form.get('father_occupation')
    admission_grade = request.form.get('admission_grade')
    displaced = request.form.get('displaced')
    educational_special_needs = request.form.get('educational_special_needs')
    debtor = request.form.get('debtor')
    tuition_fees_up_to_date = request.form.get('tuition_fees_up_to_date')
    gender = request.form.get('gender')
    scholarship_holder = request.form.get('scholarship_holder')
    age_at_enrollment = request.form.get('age_at_enrollment')
    international = request.form.get('international')
    curricular_units_1st_sem_credited = request.form.get('curricular_units_1st_sem_credited')
    curricular_units_1st_sem_enrolled = request.form.get('curricular_units_1st_sem_enrolled')
    curricular_units_1st_sem_evaluations = request.form.get('curricular_units_1st_sem_evaluations')
    curricular_units_1st_sem_approved = request.form.get('curricular_units_1st_sem_approved')
    curricular_units_1st_sem_grade = request.form.get('curricular_units_1st_sem_grade')
    curricular_units_1st_sem_without_evaluations = request.form.get('curricular_units_1st_sem_without_evaluations')
    curricular_units_2nd_sem_credited = request.form.get('curricular_units_2nd_sem_credited')
    curricular_units_2nd_sem_enrolled = request.form.get('curricular_units_2nd_sem_enrolled')
    curricular_units_2nd_sem_evaluations = request.form.get('curricular_units_2nd_sem_evaluations')
    curricular_units_2nd_sem_approved = request.form.get('curricular_units_2nd_sem_approved')
    curricular_units_2nd_sem_grade = request.form.get('curricular_units_2nd_sem_grade')
    curricular_units_2nd_sem_without_evaluations = request.form.get('curricular_units_2nd_sem_without_evaluations')
    unemployment_rate = request.form.get('unemployment_rate')
    inflation_rate = request.form.get('inflation_rate')
    gdp = request.form.get('gdp')
    
    result = model.predict([[marital_status,application_mode, application_order, course, attendance, prev_qualification, prev_qualification_grade,
                            nationality, mother_qualification, father_qualification, mother_occupation, father_occupation, 
                            admission_grade, displaced, educational_special_needs, debtor, tuition_fees_up_to_date, gender, 
                            scholarship_holder, age_at_enrollment, international, curricular_units_1st_sem_credited, 
                            curricular_units_1st_sem_enrolled, curricular_units_1st_sem_evaluations, curricular_units_1st_sem_approved,
                            curricular_units_1st_sem_grade, curricular_units_1st_sem_without_evaluations, curricular_units_2nd_sem_credited,
                            curricular_units_2nd_sem_enrolled, curricular_units_2nd_sem_evaluations, curricular_units_2nd_sem_approved,
                            curricular_units_2nd_sem_grade, curricular_units_2nd_sem_without_evaluations, unemployment_rate, inflation_rate,
                            gdp ]])[0]
    
    
    if result == 1:
        return render_template('dropout.html', prediction="The individual is predicted to be a dropout.")
    else:
        return render_template('dropout.html', prediction="The individual is predicted to not dropout.")


if __name__ == "__main__":
    app.run(debug=True)
