import sys
import os
import json
import httpx
from dotenv import load_dotenv
from google import genai
from mcp.server.fastmcp import FastMCP

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
mcp = FastMCP("MedAssist AI")

def ask_ai(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

@mcp.tool()
def symptom_analyzer(
    symptoms: str,
    patient_age: int,
    duration_days: int,
    severity_1_to_10: int,
    existing_conditions: str = "none",
    language: str = "English"
) -> str:
    """Analyze patient symptoms with step-by-step clinical reasoning. Returns differential diagnoses, risk level LOW/MEDIUM/HIGH/EMERGENCY, and recommended action."""
    prompt = f"""You are a clinical decision support AI. Think step by step carefully.

Patient: {patient_age} years old
Symptoms: {symptoms}
Duration: {duration_days} days
Severity: {severity_1_to_10}/10
Known conditions: {existing_conditions}

STEP 1 - SYMPTOM ANALYSIS:
[Analyze each symptom systematically]

STEP 2 - DIFFERENTIAL DIAGNOSIS:
1. [Most likely condition] - [probability]%
2. [Second likely condition] - [probability]%
3. [Third likely condition] - [probability]%

STEP 3 - RISK CLASSIFICATION:
Risk Level: [LOW / MEDIUM / HIGH / EMERGENCY]
Reason: [explain clearly why]

STEP 4 - RECOMMENDED ACTION:
[Home care / See clinic within 24h / Go to ER now / Call ambulance immediately]

STEP 5 - RED FLAG SYMPTOMS:
Watch for: [list warning signs]

STEP 6 - IMMEDIATE CARE ADVICE:
[Safe home care tips]

DISCLAIMER: This is AI-assisted clinical decision support only. A licensed clinician must review.

Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def image_analyzer(
    image_base64: str,
    patient_age: int,
    pain_level: int = 0,
    duration_days: int = 0,
    description: str = "",
    language: str = "English"
) -> str:
    """Analyze wound or skin condition from base64 image. Returns severity and care recommendations."""
    try:
        import PIL.Image
        import io
        import base64
        image_data = base64.b64decode(image_base64)
        image = PIL.Image.open(io.BytesIO(image_data))
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                image,
                f"""You are a clinical image analysis AI.
FIRST check: Is this a medical/health image (wound, skin, rash, injury)?
If NOT: respond only with: Please upload a clear photo of a wound or skin condition.

If YES, patient is {patient_age} years old, pain {pain_level}/10, duration {duration_days} days.
Extra info: {description}

ASSESSMENT: [What you observe]
POSSIBLE CONDITIONS: [Top 2-3]
SEVERITY: [LOW / MEDIUM / HIGH / EMERGENCY]
IMMEDIATE CARE: [What to do now]
SEE DOCTOR: [Yes urgently / Yes within 48h / Monitor at home]

DISCLAIMER: AI visual assessment only. Always seek professional medical evaluation.
Respond in {language}."""
            ]
        )
        return response.text
    except Exception as e:
        return f"Image analysis error: {str(e)}"


@mcp.tool()
def risk_calculator(
    patient_age: int,
    patient_sex: str,
    symptoms: str,
    family_history: str = "none",
    smoker: bool = False,
    diabetic: bool = False,
    hypertension: bool = False,
    language: str = "English"
) -> str:
    """Calculate age-weighted disease risk percentages. Returns top 3 conditions with percentage risk."""
    prompt = f"""You are a medical risk assessment AI.

Patient: {patient_age} years old, {patient_sex}
Symptoms: {symptoms}
Family history: {family_history}
Smoker: {smoker}, Diabetic: {diabetic}, Hypertension: {hypertension}

RISK ASSESSMENT:
Condition 1: [Name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]
Condition 2: [Name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]
Condition 3: [Name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]

OVERALL RISK LEVEL: [LOW / MEDIUM / HIGH]
WHEN TO SEE DOCTOR: [specific timeline]
AGE CONTEXT: [how age {patient_age} affects risks]
RISK REDUCTION: [3 actions]

Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def triage_classifier(
    chief_complaint: str,
    patient_age: int = 30,
    vital_signs: str = "unknown",
    consciousness_level: str = "alert",
    language: str = "English"
) -> str:
    """Classify urgency using Emergency Severity Index ESI 1-5. ESI 1=life threat, ESI 5=non-urgent."""
    prompt = f"""You are an emergency triage nurse AI using ESI protocol.

Patient: {patient_age} years old
Complaint: {chief_complaint}
Vitals: {vital_signs}
Consciousness: {consciousness_level}

ESI CLASSIFICATION: [1/2/3/4/5]

REASONING:
Step 1 - Life threat: [Yes/No + why]
Step 2 - High risk: [Yes/No + why]
Step 3 - Resources needed: [how many]
Step 4 - Final ESI: [level + full justification]

MAX WAIT TIME: [safe wait duration]
ACTION: [exact next steps]

ESI 1-2: Call 999 Malaysia immediately.
Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def report_interpreter(
    report_text: str,
    report_type: str = "lab",
    patient_age: int = 30,
    language: str = "English"
) -> str:
    """Interpret lab or radiology reports in plain language. Flags abnormal and urgent findings."""
    prompt = f"""You are a medical report interpreter.

Report type: {report_type}, Patient age: {patient_age}
Report:
---
{report_text}
---

SUMMARY: [2 simple sentences]
NORMAL FINDINGS: [what is normal]
ABNORMAL FINDINGS: [each issue, meaning, severity]
URGENT FLAGS: [anything needing immediate attention]
PLAIN LANGUAGE: [explain without jargon]
QUESTIONS FOR DOCTOR: [3-5 questions]

DISCLAIMER: Discuss all results with your doctor.
Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def drug_explainer(
    drug_name: str,
    patient_age: int = 30,
    other_medications: str = "none",
    language: str = "English"
) -> str:
    """Explain medication, dosage, side effects and check dangerous drug interactions."""
    prompt = f"""You are a clinical pharmacist AI.

Drug: {drug_name}, Patient age: {patient_age}
Other medications: {other_medications}

WHAT IT IS: [simple explanation]
HOW TO TAKE IT: dosage, timing, food, avoid
COMMON SIDE EFFECTS: [top 5]
SERIOUS SIDE EFFECTS: [dangerous reactions - seek help]
DRUG INTERACTIONS: [check {drug_name} vs {other_medications}]
MISSED DOSE: [what to do]
STORAGE: [how to store]

DISCLAIMER: Follow your doctor's prescription.
Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def nearby_facilities(
    latitude: float,
    longitude: float,
    urgency_level: str = "MEDIUM",
    language: str = "English"
) -> str:
    """Find nearby hospitals or clinics. EMERGENCY returns ambulance numbers immediately."""
    if urgency_level == "EMERGENCY":
        return """EMERGENCY - CALL AMBULANCE NOW
MALAYSIA: 999 or 112
SINGAPORE: 995
INTERNATIONAL: 112
Do NOT wait. Call now and go to nearest ER."""

    try:
        facility_type = "hospital" if urgency_level == "HIGH" else "clinic"
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": facility_type,
            "format": "json",
            "limit": 5,
            "viewbox": f"{longitude-0.1},{latitude+0.1},{longitude+0.1},{latitude-0.1}",
            "bounded": 1
        }
        headers = {"User-Agent": "MedAssist-AI/1.0"}
        r = httpx.get(url, params=params, headers=headers, timeout=10)
        results = r.json()
        if results:
            output = f"Nearby {facility_type}s:\n\n"
            for i, place in enumerate(results[:4], 1):
                output += f"{i}. {place.get('display_name', 'Unknown')}\n\n"
            output += f"\nMap: https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}&zoom=14"
            output += "\nEmergency: 999 (Malaysia)"
            return output
        return f"No results. Search Google Maps for '{facility_type} near me'. Emergency: 999"
    except Exception as e:
        return "Location search failed. Search Google Maps for 'hospital near me'. Emergency: 999"


@mcp.tool()
def mental_health_screener(
    screen_type: str = "PHQ9",
    patient_responses: str = "",
    patient_age: int = 25,
    language: str = "English"
) -> str:
    """Run PHQ-9 depression or GAD-7 anxiety screening with scoring and supportive guidance."""
    prompt = f"""You are a compassionate mental health screening AI.

Screen: {screen_type}, Patient age: {patient_age}
Responses: {patient_responses if patient_responses else "none - provide the questions"}

If NO responses: give full questionnaire warmly. Rate: 0=Not at all, 1=Several days, 2=More than half days, 3=Nearly every day.

If responses provided:
PHQ-9: 0-4=Minimal, 5-9=Mild, 10-14=Moderate, 15-19=Mod.Severe, 20-27=Severe
GAD-7: 0-4=Minimal, 5-9=Mild, 10-14=Moderate, 15-21=Severe

SCORE: [X/max]
SEVERITY: [level]
INTERPRETATION: [warm, non-judgmental explanation]
SUPPORT: [actionable steps]
RESOURCES Malaysia:
- Befrienders KL: 03-7627 2929 (24h)
- Mental Health Support: 03-2935 9935

This is not a diagnosis. Please see a mental health professional.
Respond in {language} with empathy."""
    return ask_ai(prompt)


@mcp.tool()
def fhir_patient_data(
    patient_id: str,
    fhir_base_url: str = "https://r4.smarthealthit.org",
    language: str = "English"
) -> str:
    """Retrieve and summarize patient data from FHIR R4 server."""
    try:
        headers = {"Accept": "application/fhir+json"}
        summary = {"patient_id": patient_id}

        r = httpx.get(f"{fhir_base_url}/Patient/{patient_id}", headers=headers, timeout=10)
        if r.status_code == 200:
            p = r.json()
            name_obj = p.get("name", [{}])[0]
            summary["name"] = f"{name_obj.get('given', [''])[0]} {name_obj.get('family', '')}".strip()
            summary["birthDate"] = p.get("birthDate", "unknown")
            summary["gender"] = p.get("gender", "unknown")

        r2 = httpx.get(f"{fhir_base_url}/Condition?patient={patient_id}&_count=10", headers=headers, timeout=10)
        if r2.status_code == 200:
            summary["conditions"] = [e["resource"].get("code", {}).get("text", "unknown") for e in r2.json().get("entry", [])]

        r3 = httpx.get(f"{fhir_base_url}/MedicationRequest?patient={patient_id}&_count=10", headers=headers, timeout=10)
        if r3.status_code == 200:
            summary["medications"] = [e["resource"].get("medicationCodeableConcept", {}).get("text", "unknown") for e in r3.json().get("entry", [])]

        prompt = f"""Summarize this FHIR patient record:
{json.dumps(summary, indent=2)}

PATIENT SUMMARY: [Name, age, gender]
ACTIVE CONDITIONS: [list]
CURRENT MEDICATIONS: [list]
CLINICAL NOTES: [important observations]

Respond in {language}."""
        return ask_ai(prompt)
    except Exception as e:
        return f"Could not retrieve FHIR data: {str(e)}"


@mcp.tool()
def guidelines_lookup(
    condition_or_question: str,
    clinician_type: str = "general practitioner",
    language: str = "English"
) -> str:
    """Evidence-based clinical guidelines from WHO, NICE, CDC, Malaysian CPG."""
    prompt = f"""You are a clinical guidelines AI for {clinician_type}.

Question: {condition_or_question}

GUIDELINE SUMMARY: [WHO/NICE/CDC/Malaysian CPG recommendations]
DIAGNOSTIC CRITERIA: [how to diagnose]
FIRST-LINE TREATMENT: [evidence-based approach]
WHEN TO REFER: [specialist referral criteria]
MONITORING: [what to track, how often]
EVIDENCE GRADE: [A/B/C]
SOURCE: [organization]

Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def soap_note_generator(
    patient_age: int,
    chief_complaint: str,
    symptoms: str,
    assessment: str,
    plan_notes: str = "",
    language: str = "English"
) -> str:
    """Generate structured SOAP clinical note for doctor documentation."""
    prompt = f"""Generate a professional SOAP note.

Patient: {patient_age} years old
Chief complaint: {chief_complaint}
Symptoms: {symptoms}
Assessment: {assessment}
Notes: {plan_notes}

SUBJECTIVE (S): Chief complaint, history, duration, severity, associated symptoms
OBJECTIVE (O): Vitals [clinician to record], Physical exam [clinician to perform]
ASSESSMENT (A): Primary impression, differential diagnoses 1-3, risk level
PLAN (P): Investigations, treatment, medications, follow-up, patient education, referral

NOTE: Clinician examination required. This is a documentation aid only.
Respond in {language}."""
    return ask_ai(prompt)


@mcp.tool()
def patient_education(
    condition: str,
    patient_age: int = 30,
    language: str = "English"
) -> str:
    """Generate simple patient education sheet for any condition."""
    prompt = f"""Patient education sheet for {condition}, patient age {patient_age}.

WHAT IS IT: [simple explanation, no jargon]
CAUSES: [common causes simply]
SYMPTOMS: [clear list]
TREATMENT: [general approach]
HOME CARE: [practical tips]
SEE DOCTOR URGENTLY IF: [warning signs]
PREVENTION: [reduce risk tips]

Write in simple {language} for someone with no medical background."""
    return ask_ai(prompt)


if __name__ == "__main__":
    import uvicorn
    app = mcp.streamable_http_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        forwarded_allow_ips="*",
        proxy_headers=True
    )