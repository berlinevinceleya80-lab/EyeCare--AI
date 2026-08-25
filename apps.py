import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import os
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EyeCare AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = "eyecare_dr_model_v1.keras"

CLASS_NAMES = [
    "No DR",
    "NPDR",
    "PDR"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f8fc;
}

.main-header {
    background: linear-gradient(
        135deg,
        #063b52,
        #087ea4
    );
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.main-header h1 {
    font-size: 42px;
    margin: 0;
}

.main-header p {
    font-size: 18px;
    margin-top: 8px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e1e8ef;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.section-title {
    font-size: 27px;
    font-weight: 750;
    color: #123f52;
    margin-top: 20px;
    margin-bottom: 15px;
}

.workflow-card {
    background: white;
    padding: 15px 8px;
    border-radius: 13px;
    border: 1px solid #dce6ed;
    text-align: center;
    min-height: 110px;
}

.workflow-number {
    font-size: 20px;
    font-weight: bold;
    color: #087ea4;
}

.workflow-icon {
    font-size: 27px;
}

.workflow-text {
    font-size: 13px;
    font-weight: 600;
}

.result-card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    border: 2px solid #dce6ed;
    text-align: center;
}

.footer {
    text-align: center;
    color: #687780;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "image" not in st.session_state:
    st.session_state.image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "result" not in st.session_state:
    st.session_state.result = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "screened" not in st.session_state:
    st.session_state.screened = False


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {os.path.abspath(MODEL_PATH)}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


try:
    model = load_model()
    model_status = True
    model_error = ""

except Exception as e:
    model = None
    model_status = False
    model_error = str(e)

# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf(
    patient_id,
    patient_name,
    patient_age,
    diabetes_duration,
    previous_screening,
    eye,
    camp_name,
    camp_location,
    quality_status,
    result,
    confidence,
    predictions,
    referral
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=15,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=7
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "EYECARE AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Diabetic Retinopathy Screening Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "Portable AI-Assisted Screening for Rural Eye Camps",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Patient Information",
            heading_style
        )
    )

    patient_data = [
        ["Patient ID", patient_id or "Not provided"],
        ["Patient Name", patient_name or "Not provided"],
        ["Age", str(patient_age)],
        ["Diabetes Duration", f"{diabetes_duration} years"],
        ["Eye", eye],
        ["Previous Screening", previous_screening]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[55 * mm, 110 * mm]
    )

    patient_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4f8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(patient_table)

    # --------------------------------------------------------
    # CAMP INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Eye Camp Information",
            heading_style
        )
    )

    camp_data = [
        ["Camp Name", camp_name or "Not provided"],
        ["Camp Location", camp_location or "Not provided"],
        [
            "Screening Date & Time",
            datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        ]
    ]

    camp_table = Table(
        camp_data,
        colWidths=[55 * mm, 110 * mm]
    )

    camp_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4f8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(camp_table)

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Image Quality Assessment",
            heading_style
        )
    )

    quality_data = [
        ["Image Quality", quality_status],
        ["AI Input Size", "224 × 224 pixels"],
        ["Image Type", "Retinal Fundus Image"]
    ]

    quality_table = Table(
        quality_data,
        colWidths=[55 * mm, 110 * mm]
    )

    quality_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4f8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(quality_table)

    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. AI Analysis",
            heading_style
        )
    )

    ai_data = [
        ["AI Classification", result],
        [
            "AI Confidence",
            f"{confidence * 100:.2f}%"
        ],
        [
            "Model",
            "EyeCare AI Deep Learning Model"
        ]
    ]

    ai_table = Table(
        ai_data,
        colWidths=[55 * mm, 110 * mm]
    )

    ai_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4f8")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(ai_table)

    # --------------------------------------------------------
    # AI PROBABILITIES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. AI Probability Distribution",
            heading_style
        )
    )

    probability_data = [
        ["Class", "Probability"]
    ]

    for i, class_name in enumerate(CLASS_NAMES):

        probability_data.append([
            class_name,
            f"{predictions[i] * 100:.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[100 * mm, 65 * mm]
    )

    probability_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087ea4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(probability_table)

    # --------------------------------------------------------
    # REFERRAL
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Referral Recommendation",
            heading_style
        )
    )

    referral_table = Table(
        [[
            Paragraph(
                referral,
                normal_style
            )
        ]],
        colWidths=[165 * mm]
    )

    referral_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#087ea4")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f8fb")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(referral_table)

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "7. Important Disclaimer",
            heading_style
        )
    )

    disclaimer = """
    This report provides an AI-assisted preliminary screening result.
    It is not a medical diagnosis. The result should be reviewed
    and confirmed by a qualified ophthalmologist or other
    appropriate healthcare professional.
    """

    story.append(
        Paragraph(
            disclaimer,
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Generated by EyeCare AI",
            subtitle_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">

<h1>👁️ EyeCare AI</h1>

<p>
Portable AI-Assisted Diabetic Retinopathy Screening
for Rural Eye Camps
</p>

</div>
""", unsafe_allow_html=True)

st.write(
    "An AI-assisted screening platform designed to help "
    "healthcare workers identify retinal images that may "
    "require further ophthalmological evaluation."
)

st.warning(
    "⚠️ AI-assisted screening only. "
    "This system is not a replacement for professional "
    "medical diagnosis."
)


# ============================================================
# SYSTEM STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    if model_status:
        st.success("🟢 AI Model Online")
    else:
        st.error("🔴 AI Model Offline")

with col2:

    st.metric(
        "AI Classes",
        "3"
    )

with col3:

    st.metric(
        "Input Size",
        "224 × 224"
    )

with col4:

    st.metric(
        "Mode",
        "Screening"
    )


# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    '<div class="section-title">🔄 Screening Workflow</div>',
    unsafe_allow_html=True
)

workflow = [
    ("01", "👤", "Patient", "Registration"),
    ("02", "📷", "Fundus", "Capture"),
    ("03", "🔍", "Image", "Quality"),
    ("04", "🧠", "AI", "Analysis"),
    ("05", "📊", "Screening", "Result"),
    ("06", "🏥", "Referral", "Decision"),
    ("07", "📄", "Digital", "Report")
]

workflow_columns = st.columns(7)

for i, item in enumerate(workflow):

    number, icon, title, subtitle = item

    with workflow_columns[i]:

        st.markdown(
            f"""
            <div class="workflow-card">

            <div class="workflow-number">
            {number}
            </div>

            <div class="workflow-icon">
            {icon}
            </div>

            <div class="workflow-text">
            {title}<br>{subtitle}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 EyeCare AI")

    st.subheader("Eye Camp Details")

    camp_name = st.text_input(
        "Camp Name",
        value="Rural Eye Screening Camp"
    )

    camp_location = st.text_input(
        "Camp Location"
    )

    st.divider()

    st.subheader("🧠 AI Model")

    if model_status:

        st.success("Model Loaded")

    else:

        st.error("Model Not Found")

    st.write("Model: MobileNetV2")

    st.write("Classes:")
    st.write("• No DR")
    st.write("• NPDR")
    st.write("• PDR")


# ============================================================
# PATIENT REGISTRATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '01 — 👤 Patient Registration'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    patient_id = st.text_input(
        "Patient ID",
        placeholder="EC-001"
    )

with col2:

    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter patient name"
    )

with col3:

    patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

col1, col2, col3 = st.columns(3)

with col1:

    diabetes_duration = st.number_input(
        "Diabetes Duration (Years)",
        min_value=0,
        max_value=100,
        value=0
    )

with col2:

    previous_screening = st.selectbox(
        "Previous Screening",
        [
            "Not Done",
            "Normal",
            "DR Detected",
            "Unknown"
        ]
    )

with col3:

    eye = st.selectbox(
        "Eye",
        [
            "Right Eye",
            "Left Eye",
            "Both Eyes"
        ]
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CAMERA / UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '02 — 📷 Fundus Image Capture'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Capture a retinal fundus image using your camera "
    "or upload an existing fundus image."
)

tab_camera, tab_upload = st.tabs(
    [
        "📷 Camera",
        "📁 Upload Image"
    ]
)

with tab_camera:

    camera_file = st.camera_input(
        "Capture retinal image"
    )

    if camera_file is not None:

        st.session_state.image = (
            Image.open(camera_file)
            .convert("RGB")
        )

with tab_upload:

    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.session_state.image = (
            Image.open(uploaded_file)
            .convert("RGB")
        )


# ============================================================
# DISPLAY IMAGE
# ============================================================

if st.session_state.image is not None:

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.image(
            st.session_state.image,
            caption="Retinal Fundus Image",
            use_container_width=True
        )

    with col2:

        width, height = (
            st.session_state.image.size
        )

        st.subheader(
            "📋 Image Information"
        )

        st.write(
            f"**Width:** {width}px"
        )

        st.write(
            f"**Height:** {height}px"
        )

        st.write(
            "**Format:** RGB"
        )

        st.success(
            "Image successfully loaded."
        )

else:

    st.info(
        "Please capture or upload a fundus image."
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '03 — 🔍 Image Quality Check'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.image is not None:

    img_array = np.array(
        st.session_state.image
    )

    width, height = (
        st.session_state.image.size
    )

    brightness = float(
        np.mean(img_array)
    )

    contrast = float(
        np.std(img_array)
    )

    if (
        width >= 224
        and height >= 224
        and contrast >= 20
    ):

        quality_status = "GOOD"

        st.success(
            "🟢 Image quality appears suitable "
            "for AI screening."
        )

    else:

        quality_status = "POOR"

        st.warning(
            "🟡 Image quality may affect AI performance."
        )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.metric(
            "Resolution",
            f"{width} × {height}"
        )

    with q2:
        st.metric(
            "Brightness",
            f"{brightness:.1f}"
        )

    with q3:
        st.metric(
            "Contrast",
            f"{contrast:.1f}"
        )

else:

    quality_status = "NOT CHECKED"

    st.info(
        "Upload an image to perform quality check."
    )


# ============================================================
# AI SCREENING
# ============================================================

st.markdown(
    '<div class="section-title">'
    '04 — 🧠 AI Retinal Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "The trained deep-learning model processes the retinal "
    "image and estimates the probability of No DR, NPDR "
    "and PDR."
)

if st.button(
    "🚀 START AI SCREENING",
    type="primary",
    use_container_width=True
):

    if st.session_state.image is None:

        st.error(
            "❌ Please capture or upload a retinal image first."
        )

    elif not model_status:

        st.error(
            "❌ AI model could not be loaded."
        )

    else:

        with st.spinner(
            "🧠 AI is analyzing the retinal image..."
        ):

            image = (
                st.session_state.image
                .resize((224, 224))
            )

            image_array = np.array(
                image
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = np.array(image).astype("float32")

            image_array = image_array / 255.0

            image_array = np.expand_dims(
            image_array,
            axis=0
            )
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = (
                CLASS_NAMES[predicted_index]
            )

            confidence = float(
                predictions[predicted_index]
            )

            st.session_state.prediction = predictions
            st.session_state.result = predicted_class
            st.session_state.confidence = confidence
            st.session_state.screened = True

        st.success(
            "✅ AI screening completed successfully."
        )


# ============================================================
# AI RESULT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '05 — 📊 AI Screening Result'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.screened:

    result = st.session_state.result
    confidence = st.session_state.confidence
    predictions = st.session_state.prediction

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "AI Classification"
    )

    if result == "No DR":

        st.success(
            f"🟢 NO DR\n\n"
            f"Confidence: {confidence * 100:.2f}%"
        )

    elif result == "NPDR":

        st.warning(
            f"🟡 NPDR\n\n"
            f"Confidence: {confidence * 100:.2f}%"
        )

    else:

        st.error(
            f"🔴 PDR\n\n"
            f"Confidence: {confidence * 100:.2f}%"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🧠 AI Probability Distribution"
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = float(
            predictions[i]
        )

        st.write(
            f"**{class_name} — "
            f"{probability * 100:.2f}%**"
        )

        st.progress(
            probability
        )

    st.subheader(
        "📌 AI Interpretation"
    )

    if result == "No DR":

        st.success(
            "No obvious diabetic retinopathy pattern "
            "was detected by the AI screening model."
        )

    elif result == "NPDR":

        st.warning(
            "The AI detected a retinal pattern "
            "consistent with Non-Proliferative "
            "Diabetic Retinopathy."
        )

    else:

        st.error(
            "The AI detected a retinal pattern "
            "consistent with Proliferative "
            "Diabetic Retinopathy."
        )

else:

    st.info(
        "AI result will appear here after screening."
    )


# ============================================================
# REFERRAL
# ============================================================

st.markdown(
    '<div class="section-title">'
    '06 — 🏥 Referral Recommendation'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.screened:

    result = st.session_state.result

    if result == "No DR":

        referral = (
            "Routine eye screening and periodic "
            "follow-up are recommended."
        )

        st.success(
            "🟢 ROUTINE FOLLOW-UP"
        )

    elif result == "NPDR":

        referral = (
            "Ophthalmologist review is recommended "
            "for further assessment."
        )

        st.warning(
            "🟡 OPHTHALMOLOGIST REVIEW"
        )

    else:

        referral = (
            "Priority ophthalmologist evaluation "
            "is recommended."
        )

        st.error(
            "🔴 PRIORITY REFERRAL"
        )

    st.write(
        referral
    )

else:

    referral = (
        "Complete AI screening first."
    )

    st.info(
        referral
    )


# ============================================================
# PDF REPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '07 — 📄 Screening Report'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.screened:

    pdf_data = generate_pdf(
        patient_id,
        patient_name,
        patient_age,
        diabetes_duration,
        previous_screening,
        eye,
        camp_name,
        camp_location,
        quality_status,
        st.session_state.result,
        st.session_state.confidence,
        st.session_state.prediction,
        referral
    )

    st.success(
        "✅ Professional screening report generated."
    )

    st.download_button(
        label="📥 DOWNLOAD SCREENING REPORT — PDF",
        data=pdf_data,
        file_name=(
            f"{patient_id or 'Patient'}"
            "_EyeCare_AI_Report.pdf"
        ),
        mime="application/pdf",
        use_container_width=True
    )

else:

    st.info(
        "Complete AI screening to generate the PDF report."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
<div class="footer">

<b>👁️ EyeCare AI</b><br>

Portable AI-Assisted Diabetic Retinopathy Screening
for Rural Eye Camps

<br><br>

AI-assisted screening •
Not a replacement for professional diagnosis

</div>
""",
    unsafe_allow_html=True
)
