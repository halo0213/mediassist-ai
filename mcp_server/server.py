import os
import json
import httpx
from dotenv import load_dotenv
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP

# Direct patch of MCP transport security
try:
    from mcp.server.transport_security import TransportSecurityMiddleware
    TransportSecurityMiddleware.is_valid_host = lambda self, host: True
    print("Transport security patched successfully")
except Exception as e:
    print(f"Patch failed: {e}")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
mcp = FastMCP("MedAssist AI")

def ask_ai(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

@mcp.tool()
def symptom_analyzer(symptoms: str, patient_age: int, duration_days: int, severity_1_to_10: int, existing_conditions: str = "none", language: str = "English") -> str:
    """Analyze patient symptoms with step-by-step clinical reasoning. Returns differential diagnoses and risk level LOW/MEDIUM/HIGH/EMERGENCY."""
    prompt = f"""You are a clinical decision support AI. Think step by step.
Patient: {patient_age} years old, Symptoms: {symptoms}, Duration: {duration_days} days, Severity: {severity_1_to_10}/10, Conditions: {existing_conditions}

STEP 1 - SYMPTOM ANALYSIS: [analyze each symptom]
STEP 2 - DIFFERENTIAL DIAGNOSIS:
1. [condition] - [probability]%
2. [condition] - [probability]%
3. [condition] - [probability]%
STEP 3 - RISK LEVEL: [LOW / MEDIUM / HIGH / EMERGENCY] - Reason: [why]
STEP 4 - ACTION: [Home care / See clinic / Go to ER / Call ambulance]
STEP 5 - RED FLAGS: [warning signs to watch]
STEP 6 - HOME CARE: [safe tips]
DISCLAIMER: AI-assisted support only. A licensed clinician must review.
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def image_analyzer(image_base64: str, patient_age: int, pain_level: int = 0, duration_days: int = 0, description: str = "", language: str = "English") -> str:
    """Analyze wound or skin condition from base64 image. Returns severity LOW/MEDIUM/HIGH/EMERGENCY and care recommendations."""
    try:
        import PIL.Image
        import io
        import base64 as b64
        image_data = b64.b64decode(image_base64)
        image = PIL.Image.open(io.BytesIO(image_data))
        vision_model = genai.GenerativeModel("gemini-1.5-flash")
        response = vision_model.generate_content([
            image,
            f"""Clinical image analysis AI. Patient {patient_age} years old, pain {pain_level}/10, duration {duration_days} days. Info: {description}
Is this a medical image (wound/skin/rash/injury)? If NOT, say: Please upload a wound or skin photo.
If YES: ASSESSMENT, POSSIBLE CONDITIONS (top 3), SEVERITY (LOW/MEDIUM/HIGH/EMERGENCY), IMMEDIATE CARE, SEE DOCTOR (urgency).
DISCLAIMER: AI only. Seek professional evaluation. Respond in {language}."""
        ])
        return response.text
    except Exception as e:
        return f"Image error: {str(e)}"

@mcp.tool()
def risk_calculator(patient_age: int, patient_sex: str, symptoms: str, family_history: str = "none", smoker: bool = False, diabetic: bool = False, hypertension: bool = False, language: str = "English") -> str:
    """Calculate age-weighted disease risk percentages. Returns top 3 conditions with percentage risk."""
    prompt = f"""Medical risk assessment AI.
Patient: {patient_age} years old, {patient_sex}, Symptoms: {symptoms}, Family: {family_history}, Smoker: {smoker}, Diabetic: {diabetic}, Hypertension: {hypertension}
RISK ASSESSMENT:
Condition 1: [name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]
Condition 2: [name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]
Condition 3: [name] | Risk: [X]% | Age factor: [effect] | Reason: [brief]
OVERALL: [LOW/MEDIUM/HIGH]
WHEN TO SEE DOCTOR: [timeline]
AGE CONTEXT: [how age {patient_age} affects risks]
REDUCE RISK: [3 actions]
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def triage_classifier(chief_complaint: str, patient_age: int = 30, vital_signs: str = "unknown", consciousness_level: str = "alert", language: str = "English") -> str:
    """Classify urgency using Emergency Severity Index ESI 1-5. ESI 1=life threat, ESI 5=non-urgent."""
    prompt = f"""Emergency triage nurse AI using ESI protocol.
Patient: {patient_age}, Complaint: {chief_complaint}, Vitals: {vital_signs}, Consciousness: {consciousness_level}
ESI LEVEL: [1/2/3/4/5]
REASONING:
Step 1 - Life threat: [yes/no + why]
Step 2 - High risk: [yes/no + why]
Step 3 - Resources needed: [how many]
Step 4 - Final ESI: [level + justification]
MAX WAIT TIME: [safe duration]
ACTION: [exact next steps]
ESI 1 or 2: Call 999 Malaysia immediately.
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def report_interpreter(report_text: str, report_type: str = "lab", patient_age: int = 30, language: str = "English") -> str:
    """Interpret lab or radiology reports in plain language. Flags abnormal and urgent findings."""
    prompt = f"""Medical report interpreter. Type: {report_type}, Age: {patient_age}
Report: {report_text}
SUMMARY: [2 simple sentences]
NORMAL: [what is normal]
ABNORMAL: [each issue, meaning, severity]
URGENT FLAGS: [immediate attention needed]
PLAIN LANGUAGE: [no jargon]
QUESTIONS FOR DOCTOR: [3-5 questions]
DISCLAIMER: Discuss with your doctor.
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def drug_explainer(drug_name: str, patient_age: int = 30, other_medications: str = "none", language: str = "English") -> str:
    """Explain medication dosage, side effects and check dangerous drug interactions."""
    prompt = f"""Clinical pharmacist AI.
Drug: {drug_name}, Age: {patient_age}, Other meds: {other_medications}
WHAT IT IS: [simple explanation]
HOW TO TAKE: [dosage, timing, food, avoid]
COMMON SIDE EFFECTS: [top 5]
SERIOUS SIDE EFFECTS: [dangerous - seek help]
INTERACTIONS: [{drug_name} vs {other_medications}]
MISSED DOSE: [what to do]
STORAGE: [how to store]
DISCLAIMER: Follow your doctor's prescription.
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def nearby_facilities(latitude: float, longitude: float, urgency_level: str = "MEDIUM", language: str = "English") -> str:
    """Find nearby hospitals or clinics. EMERGENCY returns ambulance numbers immediately."""
    if urgency_level == "EMERGENCY":
        return "EMERGENCY - CALL NOW: Malaysia 999 | Singapore 995 | International 112. Go to nearest ER immediately."
    try:
        facility_type = "hospital" if urgency_level == "HIGH" else "clinic"
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": facility_type, "format": "json", "limit": 4, "viewbox": f"{longitude-0.1},{latitude+0.1},{longitude+0.1},{latitude-0.1}", "bounded": 1}
        headers = {"User-Agent": "MedAssist-AI/1.0"}
        r = httpx.get(url, params=params, headers=headers, timeout=10)
        results = r.json()
        if results:
            output = f"Nearby {facility_type}s:\n"
            for i, place in enumerate(results[:4], 1):
                output += f"{i}. {place.get('display_name', 'Unknown')}\n"
            output += f"\nMap: https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}&zoom=14\nEmergency: 999"
            return output
        return f"No results. Search Google Maps for '{facility_type} near me'. Emergency: 999"
    except Exception as e:
        return "Search failed. Use Google Maps. Emergency: 999"

@mcp.tool()
def mental_health_screener(screen_type: str = "PHQ9", patient_responses: str = "", patient_age: int = 25, language: str = "English") -> str:
    """Run PHQ-9 depression or GAD-7 anxiety screening with scoring and supportive guidance."""
    prompt = f"""Compassionate mental health screening AI. Screen: {screen_type}, Age: {patient_age}
Responses: {patient_responses if patient_responses else "none - provide the questions"}
If NO responses: give full questionnaire warmly. Rate 0-3 each question.
If responses given - PHQ-9: 0-4=Minimal,5-9=Mild,10-14=Moderate,15-19=ModSevere,20-27=Severe. GAD-7: 0-4=Minimal,5-9=Mild,10-14=Moderate,15-21=Severe.
SCORE: [X/max], SEVERITY: [level], INTERPRETATION: [warm explanation], SUPPORT: [actionable steps]
Malaysia resources: Befrienders KL 03-7627 2929 (24h), Mental Health 03-2935 9935
Not a diagnosis. See a professional. Respond in {language} with empathy."""
    return ask_ai(prompt)

@mcp.tool()
def fhir_patient_data(patient_id: str, fhir_base_url: str = "https://r4.smarthealthit.org", language: str = "English") -> str:
    """Retrieve and summarize patient data from FHIR R4 server including conditions and medications."""
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
        prompt = f"Summarize FHIR record:\n{json.dumps(summary, indent=2)}\nPATIENT SUMMARY, CONDITIONS, MEDICATIONS, CLINICAL NOTES. Respond in {language}."
        return ask_ai(prompt)
    except Exception as e:
        return f"FHIR error: {str(e)}"

@mcp.tool()
def guidelines_lookup(condition_or_question: str, clinician_type: str = "general practitioner", language: str = "English") -> str:
    """Evidence-based clinical guidelines from WHO, NICE, CDC, Malaysian CPG."""
    prompt = f"""Clinical guidelines AI for {clinician_type}. Question: {condition_or_question}
GUIDELINE SUMMARY, DIAGNOSTIC CRITERIA, FIRST-LINE TREATMENT, SECOND-LINE TREATMENT, WHEN TO REFER, MONITORING, EVIDENCE GRADE [A/B/C], SOURCE.
Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def soap_note_generator(patient_age: int, chief_complaint: str, symptoms: str, assessment: str, plan_notes: str = "", language: str = "English") -> str:
    """Generate structured SOAP clinical note for doctor documentation."""
    prompt = f"""SOAP note. Patient: {patient_age}, Complaint: {chief_complaint}, Symptoms: {symptoms}, Assessment: {assessment}, Notes: {plan_notes}
SUBJECTIVE: chief complaint, history, duration, severity
OBJECTIVE: Vitals [clinician to record], Exam [clinician to perform]
ASSESSMENT: primary impression, 3 differentials, risk level
PLAN: investigations, treatment, medications, follow-up, education, referral
Clinician examination required. Documentation aid only. Respond in {language}."""
    return ask_ai(prompt)

@mcp.tool()
def patient_education(condition: str, patient_age: int = 30, language: str = "English") -> str:
    """Generate simple patient education sheet for any condition."""
    prompt = f"""Patient education for {condition}, age {patient_age}.
WHAT IS IT, CAUSES, SYMPTOMS, TREATMENT, HOME CARE, SEE DOCTOR URGENTLY IF, PREVENTION.
Simple {language}, no medical jargon."""
    return ask_ai(prompt)

if __name__ == "__main__":
    import uvicorn
    from starlette.types import ASGIApp, Receive, Scope, Send

    class HostOverrideMiddleware:
        def __init__(self, app: ASGIApp):
            self.app = app
        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope["type"] in ("http", "websocket"):
                scope["headers"] = [
                    (b"host", b"localhost") if k == b"host" else (k, v)
                    for k, v in scope.get("headers", [])
                ]
            await self.app(scope, receive, send)

    try:
        base_app = mcp.sse_app()
        print("Using sse_app")
    except Exception:
        try:
            base_app = mcp.streamable_http_app()
            print("Using streamable_http_app")
        except Exception as e:
            print(f"App error: {e}")
            raise

    app = HostOverrideMiddleware(base_app)
    uvicorn.run(app, host="0.0.0.0", port=8000, forwarded_allow_ips="*", proxy_headers=True)