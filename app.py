from flask import Flask, render_template, request
import numpy as np
import shap
import pandas as pd
import joblib
from flask import redirect
import plotly.express as px
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import sqlite3
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = "heartcare_secret_key"

# Load model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset
df = pd.read_csv("heart.csv")
latest_prediction = {}


def init_db():
    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()
    cursor.execute("""
CREATE TABLE IF NOT EXISTS users(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT UNIQUE,

    password TEXT

    )
""")
    

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        age REAL,

        gender TEXT,

        cholesterol REAL,

        bp REAL,

        prediction TEXT,

        confidence REAL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)

    conn.commit()

    conn.close()


init_db()

def apply_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=14),
        title_font=dict(size=20, color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


@app.route("/")
def home():
    return render_template(
        "index.html",
        active_page="prediction"
    )

@app.route("/dashboard")

def dashboard():

    total = len(df)
    disease = len(df[df["target"] == 1])
    healthy = len(df[df["target"] == 0])

    # Pie Chart
    pie = px.pie(
        df,
        names="target",
        hole=0.45,
        title="Heart Disease Distribution"
    )

    pie = apply_dark_theme(pie)

    pie_chart = pie.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    # Age Histogram
    age_fig = px.histogram(
        df,
        x="age",
        color="target",
        nbins=20,
        title="Age Distribution by Disease Status"
    )

    age_fig = apply_dark_theme(age_fig)

    age_chart = age_fig.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    # Feature Importance
    features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    imp_df = imp_df.sort_values(
        by="Importance",
        ascending=True
    )

    imp_fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance Analysis"
    )

    imp_fig = apply_dark_theme(imp_fig)

    importance_chart = imp_fig.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    return render_template(
    "dashboard.html",
    active_page="dashboard",
    total=total,
    disease=disease,
    healthy=healthy,
    pie_chart=pie_chart,
    age_chart=age_chart,
    importance_chart=importance_chart
)

@app.route("/analytics")
def analytics():

    corr = df.corr(numeric_only=True)

    heatmap = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu"
    )

    heatmap = apply_dark_theme(heatmap)

    heatmap_chart = heatmap.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    age_risk = px.box(
        df,
        x="target",
        y="age",
        color="target",
        title="Age Distribution vs Heart Disease"
    )

    age_risk = apply_dark_theme(age_risk)

    age_chart = age_risk.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    chol_chart_fig = px.histogram(
        df,
        x="chol",
        color="target",
        nbins=30,
        title="Cholesterol Analysis"
    )

    chol_chart_fig = apply_dark_theme(chol_chart_fig)

    chol_chart = chol_chart_fig.to_html(
        full_html=False,
        config={"displayModeBar": False}
    )

    return render_template(
    "analytics.html",
    active_page="analytics",
    heatmap_chart=heatmap_chart,
    age_chart=age_chart,
    chol_chart=chol_chart
)

@app.route("/reports")

def reports():

    total = len(df)

    disease = len(df[df["target"] == 1])

    healthy = len(df[df["target"] == 0])

    return render_template(
        "reports.html",
        active_page="reports",
        total=total,
        disease=disease,
        healthy=healthy
    )

@app.route("/download-pdf")
def download_pdf():

    global latest_prediction

    from datetime import datetime
    import io
    from flask import send_file
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    report_id = datetime.now().strftime(
        "HD-%Y%m%d-%H%M%S"
    )

    generated_time = datetime.now().strftime(
        "%d-%b-%Y %I:%M %p"
    )

    content = []

    if latest_prediction:

        confidence = latest_prediction["confidence"]

        if confidence < 50:
            risk_level = "LOW RISK"
        elif confidence < 75:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "HIGH RISK"

        content.append(
            Paragraph(
                "HEARTCARE AI",
                styles["Title"]
            )
        )

        content.append(
            Paragraph(
                "Heart Disease Prediction Report",
                styles["Heading2"]
            )
        )

        content.append(Spacer(1, 12))

        content.append(
            Paragraph(
                f"<b>Report ID:</b> {report_id}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"<b>Generated On:</b> {generated_time}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 15))

        content.append(
            Paragraph(
                "<b>PATIENT INFORMATION</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                f"Age: {latest_prediction['age']} Years",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Gender: {latest_prediction['sex']}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                "<b>CLINICAL PARAMETERS</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                f"Blood Pressure: {latest_prediction['bp']} mmHg",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Cholesterol: {latest_prediction['chol']} mg/dL",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                "<b>PREDICTION RESULT</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                f"Status: {latest_prediction['prediction']}",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Confidence Score: {confidence:.2f}%",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                f"Risk Level: {risk_level}",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                "<b>AI INTERPRETATION</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                "This prediction was generated using a trained machine learning "
                "model based on the supplied patient parameters.",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                "<b>RECOMMENDATIONS</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                "• Maintain a balanced diet",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                "• Exercise regularly",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                "• Monitor blood pressure and cholesterol",
                styles["Normal"]
            )
        )

        content.append(
            Paragraph(
                "• Schedule periodic cardiovascular screening",
                styles["Normal"]
            )
        )

        content.append(Spacer(1, 10))

        content.append(
            Paragraph(
                "<b>DISCLAIMER</b>",
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                "This report is generated by an AI model and should not replace "
                "professional medical diagnosis or consultation.",
                styles["Normal"]
            )
        )

    else:

        content.append(
            Paragraph(
                "No prediction available.",
                styles["Title"]
            )
        )

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="HeartCare_Report.pdf",
        mimetype="application/pdf"
    )

@app.route("/export-analytics")
def export_analytics():

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "Analytics Report",
            styles['Title']
        )
    )

    content.append(
        Paragraph(
            "Heart Disease Dataset Analytics",
            styles['Normal']
        )
    )

    content.append(
        Paragraph(
            f"Total Records: {len(df)}",
            styles['Normal']
        )
    )

    content.append(
        Paragraph(
            f"Disease Cases: {len(df[df['target']==1])}",
            styles['Normal']
        )
    )

    content.append(
        Paragraph(
            f"Healthy Cases: {len(df[df['target']==0])}",
            styles['Normal']
        )
    )

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Analytics_Report.pdf",
        mimetype="application/pdf"
    )

@app.route("/export-model")
def export_model():

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    features = [
        "age","sex","cp","trestbps","chol",
        "fbs","restecg","thalach",
        "exang","oldpeak","slope",
        "ca","thal"
    ]

    content = []

    content.append(
        Paragraph(
            "Model Performance Report",
            styles['Title']
        )
    )

    content.append(
        Paragraph(
            "Random Forest Model",
            styles['Heading2']
        )
    )

    content.append(
        Paragraph(
            "Estimated Accuracy: 98%",
            styles['Normal']
        )
    )

    content.append(Spacer(1,20))

    for feature, importance in zip(
        features,
        model.feature_importances_
    ):

        content.append(
            Paragraph(
                f"{feature}: {importance:.4f}",
                styles['Normal']
            )
        )

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Model_Report.pdf",
        mimetype="application/pdf"
    )

@app.route("/history")

def history():

    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        active_page="history",
        rows=rows
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:

        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        patient = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        patient_scaled = scaler.transform(patient)

        prediction = model.predict(patient_scaled)[0]
        

        feature_names = [
             "Age",
            "Sex",
            "Chest Pain",
            "Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Rest ECG",
            "Max Heart Rate",
             "Exercise Angina",
             "Old Peak",
            "Slope",
            "CA",
            "Thal"
        ]

        importances = model.feature_importances_

        top_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
             reverse=True
        )[:5]

        probability = model.predict_proba(patient_scaled)

        if prediction == 1:

            confidence = probability[0][1] * 100

            result = (
                f"Heart Disease Detected "
                f"({confidence:.2f}% confidence)"
            )

        else:

            confidence = probability[0][0] * 100

            result = (
                f"No Heart Disease Detected "
                f"({confidence:.2f}% confidence)"
            )

        global latest_prediction

        latest_prediction = {

            "age": age,

            "sex": "Male" if sex == 1 else "Female",

            "chol": chol,

            "bp": trestbps,

            "prediction": result,

            "confidence": confidence

        }

        # SAVE TO SQLITE

        conn = sqlite3.connect("database.db")

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO predictions
        (
            age,
            gender,
            cholesterol,
            bp,
            prediction,
            confidence
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            age,
            "Male" if sex == 1 else "Female",
            chol,
            trestbps,
            result,
            confidence
        ))

        conn.commit()

        conn.close()

        return render_template(
    "index.html",
    prediction_text=result,
    top_features=top_features
)

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )
    
@app.route("/patient-report/<int:id>")
def patient_report(id):

    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        WHERE id=?
        """,
        (id,)
    )

    patient = cursor.fetchone()

    conn.close()

    if not patient:
        return "Patient not found"

    age = patient[1]
    gender = patient[2]
    chol = patient[3]
    bp = patient[4]
    prediction = patient[5]
    confidence = patient[6]

    if confidence < 50:
        risk = "LOW RISK"
    elif confidence < 75:
        risk = "MODERATE RISK"
    else:
        risk = "HIGH RISK"

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "HEARTCARE AI",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            f"Patient Report #{id}",
            styles["Heading2"]
        )
    )

    content.append(Spacer(1, 20))

    data = [
        ["Age", age],
        ["Gender", gender],
        ["Blood Pressure", f"{bp} mmHg"],
        ["Cholesterol", f"{chol} mg/dL"],
        ["Prediction", prediction],
        ["Confidence", f"{confidence:.2f}%"],
        ["Risk Level", risk]
    ]

    table = Table(data, colWidths=[180, 250])

    table.setStyle(TableStyle([
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('BACKGROUND',(0,0),(0,-1),colors.lightgrey),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold')
    ]))

    content.append(table)

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Recommendations",
            styles["Heading3"]
        )
    )

    content.append(
        Paragraph(
            "Maintain a healthy diet, exercise regularly, and schedule cardiovascular screening.",
            styles["Normal"]
        )
    )

    doc.build(content)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Patient_{id}_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__": 
    app.run(debug=True)