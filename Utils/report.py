import os
import io
import uuid
from datetime import datetime
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)

# ---------- Helper: tumor info database (expandable) ----------
_TUMOR_DB = {
    "meningioma": {
        "title": "Meningioma Tumor",
        "summary": (
            "Meningiomas are tumors that arise from the meninges, the protective layers "
            "surrounding the brain and spinal cord. Most are benign (non-cancerous)."
        ),
        "common": [
            "Grade I (Benign) - 90% of cases",
            "Grade II (Atypical) - 7–8% of cases",
            "Grade III (Malignant) - 1–3% of cases",
        ],
        "symptoms": [
            "Headaches",
            "Vision problems",
            "Hearing loss or tinnitus",
            "Memory loss",
            "Weakness in limbs",
            "Seizures"
        ],
        "treatment": [
            "Observation (for small, asymptomatic tumors)",
            "Surgical removal",
            "Stereotactic radiosurgery",
            "Conventional radiation therapy"
        ],
        "prognosis": "Generally excellent for Grade I meningiomas. Five-year survival rates are high for benign cases.",
        "prevalence": "Most common primary brain tumor, ~36% of primary brain tumors."
    },

    "glioma": {
        "title": "Glioma Tumor",
        "summary": (
            "Gliomas arise from glial cells and include a range of types (astrocytoma, oligodendroglioma, glioblastoma). "
            "They can be benign or malignant depending on grade."
        ),
        "common": [
            "Low-grade (I-II)",
            "Anaplastic (III)",
            "Glioblastoma (IV) - aggressive"
        ],
        "symptoms": [
            "Headaches",
            "Nausea/vomiting",
            "Seizures",
            "Cognitive or personality changes"
        ],
        "treatment": [
            "Surgical resection",
            "Radiotherapy",
            "Chemotherapy (temozolomide etc.)",
            "Supportive care"
        ],
        "prognosis": "Prognosis depends heavily on grade; higher-grade gliomas have worse outcomes.",
        "prevalence": "One of the most common malignant primary brain tumors in adults."
    },

    "pituitary": {
        "title": "Pituitary Tumor",
        "summary": "Pituitary tumors are growths in the pituitary gland; most are benign adenomas that affect hormone production.",
        "common": ["Non-functioning adenoma", "Prolactinoma", "Acromegaly-related pituitary adenoma"],
        "symptoms": ["Hormonal imbalance", "Vision disturbances", "Headaches"],
        "treatment": ["Surgery (transsphenoidal)", "Medical therapy", "Radiation"],
        "prognosis": "Often good with treatment; depends on subtype.",
        "prevalence": "Represents a sizeable portion of intracranial tumors (~10-15%)."
    },

    "notumor": {
        "title": "No Tumor Detected",
        "summary": "No tumor-like abnormality was detected by the AI model in the provided MRI scan.",
        "common": [],
        "symptoms": [],
        "treatment": ["No tumor-specific treatment indicated."],
        "prognosis": "N/A",
        "prevalence": "N/A"
    }
}


# ---------- Utility functions ----------
def _pil_or_array_to_tempfile(img_obj, tmp_name):
    if img_obj is None:
        return None
    if isinstance(img_obj, PILImage.Image):
        img = img_obj
    else:
        try:
            img = PILImage.fromarray(img_obj)
        except Exception:
            raise ValueError("Unsupported image format for PDF embedding.")
    img.save(tmp_name, format="PNG")
    return tmp_name


def _risk_assessment(class_label, confidence):
    conf_pct = confidence * 100 if confidence <= 1 else confidence
    try:
        conf_val = float(conf_pct)
    except Exception:
        conf_val = 0.0

    if class_label == "notumor":
        return "Healthy - No tumor detected"
    if conf_val >= 90:
        return "HIGH PRIORITY - Requires medical attention"
    if conf_val >= 75:
        return "MODERATE - Follow-up recommended"
    return "LOW - Monitor and consult if symptoms progress"


# ---------- Main PDF generator ----------
def generate_pdf_report(class_label, confidence, image, segmented_img):
    now = datetime.now()
    ts = now.strftime("%B %d, %Y at %H:%M:%S")
    report_id = f"TX-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    model_version = "TumorX v2.1.0"

    # confidence scale
    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = 0.0
    confidence_pct = confidence_val * 100 if confidence_val <= 1.05 else confidence_val

    risk_text = _risk_assessment(class_label, confidence_pct)

    out_name = f"TumorX_Report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(os.getcwd(), out_name)

    # ---------- Styles ----------
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0b3d91"),
        spaceBefore=24,   # more top spacing before each header
        spaceAfter=12
    )
    normal_style = ParagraphStyle(
        "Normal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    flow = []

    # ---------- Header (Logo + Title Block) ----------
    logo_path = None
    for candidate in ("logo.png", "tumorx_logo.png", "logo.jpg"):
        if os.path.exists(candidate):
            logo_path = candidate
            break

    if logo_path:
        rl_logo = RLImage(logo_path, width=130, height=70)
        rl_logo.hAlign = "CENTER"
        flow.append(Spacer(1, 24))
        flow.append(rl_logo)
        flow.append(Spacer(1, 18))

    flow.append(Paragraph("AI-Powered Brain Tumor Detection & Analysis", title_style))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"<b>Report Generated:</b> {ts}", subtitle_style))
    flow.append(Paragraph(f"<b>Model Version:</b> {model_version}", subtitle_style))
    flow.append(Paragraph(f"<b>Report ID:</b> {report_id}", subtitle_style))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph("<hr width='100%'/>", subtitle_style))
    flow.append(Spacer(1, 24))

    # ---------- MRI Scan Analysis ----------
    flow.append(Paragraph("MRI SCAN ANALYSIS", section_title_style))
    flow.append(Spacer(1, 12))

    tmp_files = []
    try:
        tmp_orig = f"tmp_orig_{uuid.uuid4().hex[:8]}.png"
        tmp_seg = f"tmp_seg_{uuid.uuid4().hex[:8]}.png"
        _pil_or_array_to_tempfile(image, tmp_orig)
        tmp_files.append(tmp_orig)

        if segmented_img is not None:
            _pil_or_array_to_tempfile(segmented_img, tmp_seg)
            tmp_files.append(tmp_seg)
        else:
            tmp_seg = None
    except Exception:
        tmp_orig = None
        tmp_seg = None

    imgs = []
    if tmp_orig and os.path.exists(tmp_orig):
        img1 = RLImage(tmp_orig, width=220, height=220)
        imgs.append(img1)
    else:
        imgs.append(Paragraph("Original MRI (image not available)", normal_style))

    if tmp_seg and os.path.exists(tmp_seg):
        img2 = RLImage(tmp_seg, width=220, height=220)
        imgs.append(img2)
    else:
        imgs.append(Paragraph("AI Segmentation (image not available)", normal_style))

    img_table = Table([[imgs[0], imgs[1]]], colWidths=[260, 260])
    img_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ]))
    flow.append(img_table)
    flow.append(Spacer(1, 24))

    # ---------- AI Diagnostic Results ----------
    flow.append(Paragraph("AI DIAGNOSTIC RESULTS", section_title_style))
    flow.append(Spacer(1, 12))

    diag_table_data = [
        ["Classification Result", f"{class_label}"],
        ["Confidence Level", f"{confidence_pct:.2f}%"],
        ["Risk Assessment", f"{risk_text}"]
    ]
    diag_table = Table(diag_table_data, colWidths=[160, 360])
    diag_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f3f4f6")),
    ]))
    flow.append(diag_table)
    flow.append(Spacer(1, 24))

    # ---------- Detailed Medical Info ----------
    flow.append(Paragraph("DETAILED MEDICAL INFORMATION", section_title_style))
    flow.append(Spacer(1, 12))

    info = _TUMOR_DB.get(class_label.lower(), None)
    if not info:
        flow.append(Paragraph(
            f"The AI has detected: <b>{class_label}</b>. No detailed information available.",
            normal_style
        ))
    else:
        flow.append(Paragraph(f"<b>About {info['title']}:</b>", normal_style))
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(info["summary"], normal_style))
        if info.get("common"):
            flow.append(Spacer(1,6))
            flow.append(Paragraph("<b>Common Types/Subtypes:</b>", normal_style))
            for item in info["common"]:
                flow.append(Paragraph(f"• {item}", normal_style))
        if info.get("symptoms"):
            flow.append(Spacer(1,6))
            flow.append(Paragraph("<b>Common Symptoms:</b>", normal_style))
            for item in info["symptoms"]:
                flow.append(Paragraph(f"• {item}", normal_style))
        if info.get("treatment"):
            flow.append(Spacer(1,6))
            flow.append(Paragraph("<b>Treatment Options:</b>", normal_style))
            for item in info["treatment"]:
                flow.append(Paragraph(f"• {item}", normal_style))
        if info.get("prognosis"):
            flow.append(Spacer(1,6))
            flow.append(Paragraph(f"<b>Prognosis:</b> {info['prognosis']}", normal_style))
        if info.get("prevalence"):
            flow.append(Spacer(1,6))
            flow.append(Paragraph(f"<b>Prevalence:</b> {info['prevalence']}", normal_style))

    flow.append(PageBreak())  # start next big section on new page

    # ---------- Reference Guide ----------
    flow.append(Paragraph("BRAIN TUMOR REFERENCE GUIDE", section_title_style))
    flow.append(Spacer(1, 12))
    for k, v in _TUMOR_DB.items():
        if k == "notumor":
            continue
        flow.append(Paragraph(f"<b>{v['title']}</b>", normal_style))
        flow.append(Paragraph(v["summary"], normal_style))
        flow.append(Spacer(1, 8))

    flow.append(PageBreak())

    # ---------- Disclaimers ----------
    flow.append(Paragraph("MEDICAL DISCLAIMERS & IMPORTANT INFORMATION", section_title_style))
    flow.append(Spacer(1, 12))

    disclaimers = (
        "AI Technology Limitations: This analysis is performed by artificial intelligence and machine learning algorithms. "
        "While highly accurate, AI systems can make errors and should never replace professional medical judgment.\n\n"
        "Not a Medical Diagnosis: This report provides AI-assisted analysis for informational purposes only. It does not constitute "
        "a medical diagnosis, treatment recommendation, or medical advice.\n\n"
        "Professional Medical Consultation Required: Any abnormal findings require consultation with qualified medical professionals "
        "including radiologists, neurologists, or neurosurgeons.\n\n"
        "Imaging Limitations: MRI interpretation depends on image quality, patient positioning, contrast usage, and scanning parameters. "
        "Some conditions may not be visible on MRI.\n\n"
        "Emergency Situations: If experiencing severe headaches, seizures, vision changes, or neurological symptoms, seek immediate medical attention."
    )
    for para in disclaimers.split("\n\n"):
        flow.append(Paragraph(para, normal_style))
        flow.append(Spacer(1,6))

    flow.append(Spacer(1, 24))
    flow.append(Paragraph("TumorX AI System — Advanced Brain Tumor Detection Platform", subtitle_style))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(f"Generated on {ts}", subtitle_style))

    # ---------- Build PDF ----------
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    try:
        doc.build(flow)
    finally:
        for fpath in tmp_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass

    return pdf_path
