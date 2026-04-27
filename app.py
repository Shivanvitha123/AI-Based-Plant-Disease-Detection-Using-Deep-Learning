# app.py
# AgroVision AI - FINAL TOTAL WORKING CODE
# Includes:
# ✅ Disease Detection
# ✅ Top-3 Low Confidence Selection
# ✅ Multilingual UI (English/Telugu/Hindi/Tamil)
# ✅ Translated Outputs
# ✅ Voice Output
# ✅ AI Chatbot (Groq)
# ✅ Weather API
# ✅ Disease Risk Meter
# ✅ PDF Report with Leaf Image
# ✅ Farmer Tips
# ✅ Recent Scan History

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import requests
from gtts import gTTS
import tempfile
import random
import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🌿",
    layout="wide"
)

# =====================================================
# SESSION
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# API KEYS
# =====================================================
GROQ_API_KEY = "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"
WEATHER_API_KEY = "c76874c911cb0e246cde45025c026125"


# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.stApp{background:#f5fff5;}
.block-container{padding-top:2rem;max-width:1400px;}
div.stButton > button{
border-radius:10px;
font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LANGUAGES
# =====================================================
LANG = {

"English":{
"title":"🌿 AgroVision AI",
"sub":"Smart Crop Disease Detection + Advisory",
"upload":"Upload Leaf Image",
"diagnosis":"Diagnosis",
"confidence":"Confidence",
"severity":"Severity",
"symptoms":"Symptoms",
"remedy":"Remedy",
"fertilizer":"Fertilizer",
"prevention":"Prevention",
"chat":"Ask Crop Question",
"ask":"Ask AI",
"city":"Enter City",
"check":"Check Weather",
"speak":"🔊 Speak Result",
"pdf":"📄 Download PDF Report"
},

"తెలుగు":{
"title":"🌿 అగ్రోవిజన్ AI",
"sub":"పంట వ్యాధి గుర్తింపు + సలహా",
"upload":"ఆకు చిత్రం అప్లోడ్ చేయండి",
"diagnosis":"నిర్ధారణ",
"confidence":"నమ్మకం",
"severity":"తీవ్రత",
"symptoms":"లక్షణాలు",
"remedy":"పరిష్కారం",
"fertilizer":"ఎరువు",
"prevention":"నివారణ",
"chat":"ప్రశ్న అడగండి",
"ask":"AI అడగండి",
"city":"నగరం నమోదు చేయండి",
"check":"వాతావరణం చూడండి",
"speak":"🔊 వినండి",
"pdf":"📄 PDF డౌన్‌లోడ్"
},

"हिन्दी":{
"title":"🌿 एग्रोविज़न AI",
"sub":"फसल रोग पहचान + सलाह",
"upload":"पत्ती फोटो अपलोड करें",
"diagnosis":"निदान",
"confidence":"विश्वास",
"severity":"गंभीरता",
"symptoms":"लक्षण",
"remedy":"उपाय",
"fertilizer":"उर्वरक",
"prevention":"रोकथाम",
"chat":"फसल प्रश्न पूछें",
"ask":"AI पूछें",
"city":"शहर दर्ज करें",
"check":"मौसम देखें",
"speak":"🔊 सुनें",
"pdf":"📄 PDF डाउनलोड"
},

"தமிழ்":{
"title":"🌿 அக்ரோவிஷன் AI",
"sub":"பயிர் நோய் கண்டறிதல் + ஆலோசனை",
"upload":"இலை படத்தை பதிவேற்றவும்",
"diagnosis":"கண்டறிதல்",
"confidence":"நம்பிக்கை",
"severity":"தீவிரம்",
"symptoms":"அறிகுறிகள்",
"remedy":"தீர்வு",
"fertilizer":"உரம்",
"prevention":"தடுப்பு",
"chat":"பயிர் கேள்வி கேளுங்கள்",
"ask":"AI கேளுங்கள்",
"city":"நகரம் உள்ளிடவும்",
"check":"வானிலை பார்க்கவும்",
"speak":"🔊 கேளுங்கள்",
"pdf":"📄 PDF பதிவிறக்கம்"
}
}

LANG_CODE = {
"English":"en",
"తెలుగు":"te",
"हिन्दी":"hi",
"தமிழ்":"ta"
}

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.header("⚙️ Settings")

    lang = st.selectbox("🌐 Language", list(LANG.keys()))
    T = LANG[lang]

    city_default = st.text_input("📍 City", "Hyderabad")
    threshold = st.slider("Confidence Threshold",50,95,80)

    tips = [
        "🌱 Avoid overwatering during humid days.",
        "💧 Water plants early morning.",
        "☀️ Ensure enough sunlight daily.",
        "🪴 Remove infected leaves quickly.",
        "🌾 Rotate crops regularly.",
        "🌿 Use balanced fertilizer."
    ]

    st.markdown("---")
    st.info(random.choice(tips))

    st.markdown("---")
    st.subheader("🕘 Recent Scans")

    for item in st.session_state.history[:5]:
        st.write("•", item)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("best_model.h5", compile=False)

    with open("class_indices.json","r") as f:
        labels = json.load(f)

    return model, labels

model, labels = load_assets()

# =====================================================
# TRANSLATION
# =====================================================
def translate_text(txt):

    if lang == "English":
        return txt

    LOCAL = {

        "తెలుగు":{
            "Low":"తక్కువ",
            "Moderate":"మధ్యస్థ",
            "High":"ఎక్కువ",
            "Healthy leaf":"ఆకు ఆరోగ్యంగా ఉంది",
            "Brown or black spots":"గోధుమ లేదా నలుపు మచ్చలు",
            "Circular leaf spots":"ఆకులపై వృత్తాకార మచ్చలు",
            "Use Copper spray":"కాపర్ స్ప్రే ఉపయోగించండి",
            "Use Mancozeb spray":"మాంకోజెబ్ స్ప్రే ఉపయోగించండి",
            "Balanced fertilizer":"సమతుల్య ఎరువు",
            "Potassium rich fertilizer":"పొటాషియం అధిక ఎరువు",
            "Micronutrients":"సూక్ష్మ పోషకాలు",
            "Maintain watering":"నీటిపారుదల కొనసాగించండి",
            "Avoid leaf wetness":"ఆకులు తడవకుండా చూడండి",
            "Remove infected leaves":"సంక్రమిత ఆకులను తొలగించండి",
            "Consult agriculture expert":"వ్యవసాయ నిపుణుడిని సంప్రదించండి",
            "Use healthy seeds":"ఆరోగ్యకరమైన విత్తనాలు వాడండి",
            "General crop stress":"పంట ఒత్తిడి",
            "High disease risk":"అధిక వ్యాధి ప్రమాదం"
        },

        "हिन्दी":{
            "Low":"कम",
            "Moderate":"मध्यम",
            "High":"अधिक",
            "Healthy leaf":"पत्ता स्वस्थ है",
            "Brown or black spots":"भूरे या काले धब्बे",
            "Circular leaf spots":"पत्तियों पर गोल धब्बे",
            "Use Copper spray":"कॉपर स्प्रे उपयोग करें",
            "Use Mancozeb spray":"मैनकोजेब स्प्रे उपयोग करें",
            "Balanced fertilizer":"संतुलित उर्वरक",
            "Potassium rich fertilizer":"पोटाश युक्त उर्वरक",
            "Micronutrients":"सूक्ष्म पोषक तत्व",
            "Maintain watering":"सिंचाई बनाए रखें",
            "Avoid leaf wetness":"पत्तियों को गीला न रखें",
            "Remove infected leaves":"संक्रमित पत्ते हटाएँ",
            "Consult agriculture expert":"कृषि विशेषज्ञ से संपर्क करें",
            "Use healthy seeds":"स्वस्थ बीज उपयोग करें",
            "General crop stress":"फसल तनाव",
            "High disease risk":"उच्च रोग जोखिम"
        },

        "தமிழ்":{
            "Low":"குறைவு",
            "Moderate":"மிதமான",
            "High":"அதிகம்",
            "Healthy leaf":"இலை ஆரோக்கியமாக உள்ளது",
            "Brown or black spots":"பழுப்பு அல்லது கருப்பு புள்ளிகள்",
            "Circular leaf spots":"இலைகளில் வட்டமான புள்ளிகள்",
            "Use Copper spray":"காப்பர் ஸ்ப்ரே பயன்படுத்தவும்",
            "Use Mancozeb spray":"மாங்கோசெப் தெளிக்கவும்",
            "Balanced fertilizer":"சமநிலை உரம்",
            "Potassium rich fertilizer":"பொட்டாசியம் உரம்",
            "Micronutrients":"நுண்ணூட்டச்சத்துகள்",
            "Maintain watering":"நீர்ப்பாசனம் தொடரவும்",
            "Avoid leaf wetness":"இலை நனைவதை தவிர்க்கவும்",
            "Remove infected leaves":"பாதிக்கப்பட்ட இலைகளை அகற்றவும்",
            "Consult agriculture expert":"விவசாய நிபுணரை அணுகவும்",
            "Use healthy seeds":"ஆரோக்கியமான விதைகள் பயன்படுத்தவும்",
            "General crop stress":"பயிர் அழுத்தம்",
            "High disease risk":"அதிக நோய் அபாயம்"
        }
    }

    return LOCAL.get(lang, {}).get(txt, txt)

# =====================================================
# SPEAK
# =====================================================
def speak(text):

    text = text.replace(":", "... ")
    text = text.replace(".", "... ")

    tts = gTTS(text=text, lang=LANG_CODE[lang], slow=False)

    fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(fp.name)

    audio = open(fp.name, "rb")
    st.audio(audio.read(), format="audio/mp3")

# =====================================================
# HELP DATA
# =====================================================
def disease_help(name):

    n = name.lower()

    if "healthy" in n:
        return ("Healthy leaf","No treatment needed","Balanced fertilizer","Maintain watering")

    elif "blight" in n:
        return ("Brown or black spots","Use Mancozeb spray","Potassium rich fertilizer","Avoid leaf wetness")

    elif "spot" in n:
        return ("Circular leaf spots","Use Copper spray","Micronutrients","Remove infected leaves")

    else:
        return ("General crop stress","Consult agriculture expert","Balanced fertilizer","Use healthy seeds")

# =====================================================
# PDF
# =====================================================
def create_pdf(img,disease,confidence,severity,s,r,f,p,city):

    filename = "AgroVision_Report.pdf"
    c = canvas.Canvas(filename,pagesize=A4)
    w,h = A4
    y = h - 40

    c.setFont("Helvetica-Bold",18)
    c.drawString(40,y,"AgroVision AI Report")

    y -= 40

    bio = BytesIO()
    img.save(bio,format="PNG")
    bio.seek(0)

    c.drawImage(ImageReader(bio),40,y-180,width=180,height=180)

    x = 250

    c.setFont("Helvetica",11)
    c.drawString(x,y,f"Disease: {disease}")
    y -= 20
    c.drawString(x,y,f"Confidence: {confidence:.2f}%")
    y -= 20
    c.drawString(x,y,f"Severity: {severity}")
    y -= 20
    c.drawString(x,y,f"City: {city}")
    y -= 20
    c.drawString(x,y,f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}")

    y -= 220

    c.drawString(40,y,f"Symptoms: {s}")
    y -= 20
    c.drawString(40,y,f"Remedy: {r}")
    y -= 20
    c.drawString(40,y,f"Fertilizer: {f}")
    y -= 20
    c.drawString(40,y,f"Prevention: {p}")

    c.save()
    return filename

# =====================================================
# HEADER
# =====================================================
st.title(T["title"])
st.caption(T["sub"])

tab1,tab2,tab3 = st.tabs(["🔍 Detection","🤖 AI Assistant","☁️ Weather"])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    uploaded = st.file_uploader(T["upload"], type=["jpg","jpeg","png"])

    if uploaded:

        col1,col2 = st.columns([1,2])

        with col1:
            img = Image.open(uploaded).convert("RGB")
            st.image(img,width=280)

        with col2:

            img2 = img.resize((224,224))
            arr = np.array(img2)/255.0
            arr = np.expand_dims(arr,0)

            preds = model.predict(arr,verbose=0)[0]

            top3_idx = np.argsort(preds)[-3:][::-1]
            top = top3_idx[0]

            confidence = float(preds[top])*100

            disease = labels[str(top)].replace("___"," - ").replace("_"," ")

            st.subheader(T["diagnosis"])
            st.progress(int(confidence))
            st.write(f'{T["confidence"]}: {confidence:.2f}%')

            if confidence >= threshold:
                st.success(disease)
            else:
                st.warning("Low confidence. Select correct prediction.")

                options=[]

                for i in top3_idx:
                    nm = labels[str(i)].replace("___"," - ").replace("_"," ")
                    cf = round(float(preds[i])*100,2)
                    options.append(f"{nm} ({cf}%)")

                selected = st.selectbox("Choose Prediction",options)
                disease = selected.split(" (")[0]
                st.success(disease)

            if confidence > 90:
                severity = "Low"
            elif confidence > 75:
                severity = "Moderate"
            else:
                severity = "High"

            st.write(f'{T["severity"]}: {translate_text(severity)}')

            s,r,f,p = disease_help(disease)

            st.info(f'{T["symptoms"]}: {translate_text(s)}')
            st.success(f'{T["remedy"]}: {translate_text(r)}')
            st.warning(f'{T["fertilizer"]}: {translate_text(f)}')
            st.error(f'{T["prevention"]}: {translate_text(p)}')

            item = f"{disease} ({round(confidence,1)}%)"

            if item not in st.session_state.history:
                st.session_state.history.insert(0,item)

            if st.button(T["speak"]):
                txt = f"""
{T["diagnosis"]}: {translate_text(disease)}.
{T["symptoms"]}: {translate_text(s)}.
{T["remedy"]}: {translate_text(r)}.
{T["fertilizer"]}: {translate_text(f)}.
{T["prevention"]}: {translate_text(p)}.
"""
                speak(txt)

            pdf_file = create_pdf(img,disease,confidence,severity,s,r,f,p,city_default)

            with open(pdf_file,"rb") as file:
                st.download_button(
                    T["pdf"],
                    file,
                    file_name="AgroVision_Report.pdf",
                    mime="application/pdf"
                )

# =====================================================
# TAB 2 CHATBOT
with tab2:

    q = st.text_area(T["chat"])

    if st.button(T["ask"]):

        if q.strip() == "":
            st.warning("Enter question first.")

        else:

            online = True

            try:
                requests.get("https://www.google.com", timeout=3)
            except:
                online = False

            if online and GROQ_API_KEY != "YOUR_GROQ_API_KEY":

                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":"application/json"
                }

                payload = {
                    "model":"llama-3.1-8b-instant",
                    "messages":[
                        {
                            "role":"user",
                            "content":f"""
Reply in {lang}.
You are an agriculture expert.
Give short practical farmer advice.

Question: {q}
"""
                        }
                    ]
                }

                try:

                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=20
                    )

                    if r.status_code == 200:

                        data = r.json()

                        ans = data["choices"][0]["message"]["content"]

                        st.success(ans)

                    else:
                        online = False

                except:
                    online = False

            if not online:

                q2 = q.lower()

                if "late blight" in q2:
                    st.success("Use Mancozeb spray and remove infected leaves.")

                elif "yellow leaves" in q2:
                    st.success("Nitrogen deficiency possible. Use fertilizer.")

                elif "watering" in q2:
                    st.success("Water early morning only.")

                else:
                    st.success("Check soil, pests, nutrients and irrigation.")


# =====================================================
# =====================================================
# TAB 3 WEATHER
# =====================================================
with tab3:

    city = st.text_input(T["city"], city_default)

    if st.button(T["check"]):

        try:

            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"

            r = requests.get(url,timeout=10)
            data = r.json()

            if r.status_code == 200:

                temp = data["main"]["temp"]
                hum = data["main"]["humidity"]
                desc = data["weather"][0]["description"]

                c1,c2,c3 = st.columns(3)

                c1.metric("Temp",f"{temp} °C")
                c2.metric("Humidity",f"{hum}%")
                c3.metric("Condition",desc)

                st.markdown("### 🌡 Disease Risk Meter")

                if hum > 80:
                    st.error("🔴 " + translate_text("High disease risk"))
                elif hum > 65:
                    st.warning("🟠 " + translate_text("Moderate"))
                else:
                    st.success("🟢 " + translate_text("Low"))

            else:
                st.error("Weather API Error")

        except:
            st.error("Connection Error")

st.markdown("---")
st.caption("🌿 AgroVision AI • Final Smart Agriculture Assistant")
