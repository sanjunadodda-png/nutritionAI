import streamlit as st
import base64
import os
import json
import io
import cv2
import numpy as np
import joblib
import pandas as pd
import glob
from PIL import Image
from google import genai
from dotenv import load_dotenv
from skimage.feature import hog

# ── 1. Page Configuration (Must be the very first Streamlit command) ──────────
st.set_page_config(
    page_title="NutriVision AI Portal",
    page_icon="🍎",
    layout="wide"
)

load_dotenv()

# ── 2. Initialize the login state tracker ─────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ── 3. RENDER THE LOGIN SCREEN (If not logged in yet) ─────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
        <style>
            .stApp {
                background: radial-gradient(circle at 50% 50%, #1a4243 0%, #10222a 60%, #091218 100%) !important;
            }
            [data-testid="stHeader"] {
                background: transparent !important;
            }
            .main .block-container {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                height: 100vh !important;
                min-height: 100vh !important;
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
            }
            [data-testid="stHorizontalBlock"] {
                background: #ffffff !important;
                border-radius: 20px !important;
                box-shadow: 0 35px 80px rgba(0, 0, 0, 0.6) !important;
                width: 860px !important;
                max-width: 860px !important;
                height: 530px !important;
                min-height: 530px !important;
                overflow: hidden !important;
                padding: 40px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin: 0 auto !important;
            }
            [data-testid="column"] {
                width: 50% !important;
                flex: 1 1 50% !important;
                max-width: 50% !important;
            }
            div[data-testid="stTextInput"] label {
                color: #1e293b !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                margin-bottom: 8px !important;
            }
            div[data-testid="stTextInput"] input {
                border: 1.5px solid #a7f3d0 !important;
                background-color: #f8fafc !important;
                border-radius: 10px !important;
                padding: 12px 16px !important;
                color: #1e293b !important;
            }
            div[data-testid="stTextInput"] input:focus {
                border-color: #72bf44 !important;
                box-shadow: 0 0 0 4px rgba(114, 191, 68, 0.15) !important;
            }
            div.stButton > button {
                width: 100% !important;
                background: linear-gradient(to right, #064e3b, #10b981) !important;
                color: white !important;
                border: none !important;
                border-radius: 30px !important;
                padding: 14px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.35) !important;
                margin-top: 20px !important;
                transition: opacity 0.2s ease !important;
            }
            div.stButton > button:hover {
                opacity: 0.95 !important;
                color: white !important;
            }
            .brand-title {
                font-size: 54px;
                font-weight: 700;
                margin: 0 0 5px 0;
                color: #2c4c38;
                letter-spacing: -2px;
            }
            .brand-title span {
                color: #72bf44;
            }
            .subtitle {
                color: #475569;
                font-size: 14px;
                line-height: 1.45;
                margin: 0 0 25px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.1])

    with col1:
        st.markdown("""
            <div style="text-align: center; display: flex; justify-content: center; align-items: center; height: 100%;">
                <svg viewBox="0 0 220 220" fill="none" xmlns="http://www.w3.org/2000/svg" style="width: 180px;">
                    <path d="M70 100V70C70 45 90 25 115 25C140 25 160 45 160 70V100" stroke="#166534" stroke-width="12" stroke-linecap="round"/>
                    <rect x="60" y="95" width="110" height="90" rx="18" fill="#15803d" />
                    <circle cx="115" cy="135" r="10" fill="#ffffff"/>
                </svg>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h1 class="brand-title">nutrition<span>AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter your workspace credentials to unlock tailored nutritional intelligence and analytics.</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter your name and enter the password.</p>', unsafe_allow_html=True)

        username = st.text_input("User ID", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        if st.button("Sign In to Dashboard  →"):
            if username.strip() == "admin" and password.strip() == "nutrition2026":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid User ID or Password.")
    st.stop()


# ==========================================================================
# --- 4. MAIN PROJECT DASHBOARD SYSTEM (Safely unlocked after Login) -----
# ==========================================================================

# Initialize new google-genai Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ GEMINI_API_KEY not found in environment variables or .env file!")
    st.stop()

client = genai.Client(api_key=api_key)
# Change whatever model is currently active to this explicit string:
MODEL_NAME = "gemini-3.1-flash-lite"

# Inject Application Custom Dashboard CSS Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    min-height: 100vh;
}

.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.08));
    border-radius: 20px;
    border: 1px solid rgba(16,185,129,0.15);
    margin-bottom: 2rem;
}
.hero-header h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-header p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.food-banner {
    background: linear-gradient(135deg, #065f46, #0e7490);
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    border: 1px solid rgba(16,185,129,0.3);
    margin: 1rem 0;
}
.food-banner h2 {
    color: #ecfdf5;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.food-banner span {
    color: #6ee7b7;
    font-size: 0.95rem;
}

.nutr-card {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.nutr-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(16,185,129,0.15);
}
.nutr-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #10b981;
}
.nutr-card .label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.pill {
    display: inline-block;
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #67e8f9;
    margin: 3px;
}

.health-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 12px;
    overflow: hidden;
    margin: 6px 0 16px;
}
.health-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #10b981, #06b6d4);
    transition: width 0.8s ease;
}

.benefit-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
    font-size: 0.93rem;
}
.benefit-item:last-child { border-bottom: none; }

.chat-user {
    background: linear-gradient(135deg, #065f46, #0e4f3e);
    border-radius: 16px 16px 4px 16px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    color: #ecfdf5;
    font-size: 0.95rem;
    border: 1px solid rgba(16,185,129,0.3);
}
.chat-ai {
    background: rgba(255,255,255,0.04);
    border-radius: 16px 16px 16px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 85%;
    color: #e2e8f0;
    font-size: 0.95rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.stFileUploader > div {
    border: 2px dashed rgba(16,185,129,0.3) !important;
    border-radius: 14px !important;
    background: rgba(16,185,129,0.03) !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #10b981, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Initializations ─────────────────────────────────────────────
if "food_name" not in st.session_state:
    st.session_state.food_name = None
if "nutrition" not in st.session_state:
    st.session_state.nutrition = None
if "health" not in st.session_state:
    st.session_state.health = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "history" not in st.session_state:
    st.session_state.history = []
if "confidence" not in st.session_state:
    st.session_state.confidence = None

# ── Feature Extraction & Local Predictor Setup ─────────────────────────────────
IMG_SIZE = 128
MODEL_PATH = "models/food_classifier.pkl"

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

food_model = load_classifier()

TRAINED_CLASSES = [
    "cake", "donuts", "french_fries", "fried_rice",
    "hot_dog", "ice_cream", "momos", "pizza", "waffles"
]

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys"
    )
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features = color_features / (color_features.sum() + 1e-7)
    return np.concatenate([hog_features, color_features])

def predict_food(image: Image.Image):
    if food_model is None:
        return "Model not found", None
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    features = extract_features(img_bgr).reshape(1, -1)
    label = food_model.predict(features)[0]
    confidence = None
    if hasattr(food_model, "predict_proba"):
        proba = food_model.predict_proba(features)[0]
        confidence = round(float(proba.max()) * 100, 1)
    return label, confidence

# ── CSV Dataset Intelligence Lookup ───────────────────────────────────────────
VITAMIN_COLS = ["Vitamin A","Vitamin B1","Vitamin B2","Vitamin B3","Vitamin B5","Vitamin B6","Vitamin B11","Vitamin B12","Vitamin C","Vitamin D","Vitamin E","Vitamin K"]
MINERAL_COLS = ["Calcium","Iron","Magnesium","Potassium","Zinc","Phosphorus","Copper","Manganese","Selenium"]

FOOD_KEYWORDS = {
    "waffles": ["waffle"], "pizza": ["pizza"], "hot_dog": ["hotdog", "hot dog"],
    "french_fries": ["french fries", "french frie"], "fried_rice": ["fried rice"],
    "ice_cream": ["ice cream", "gelato", "sundae"], "cake": ["cake"],
    "donuts": ["donut", "doughnut"], "momos": ["momos", "dumpling"],
    "noodles": ["noodle"], "omelette": ["omelette", "omelet", "egg"], "samosa": ["samosa"],
}

@st.cache_data
def load_nutrition_db() -> pd.DataFrame:
    files = sorted(glob.glob("data/FOOD-DATA-GROUP*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, encoding="latin1"))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def lookup_nutrition(food_class: str) -> dict | None:
    db = load_nutrition_db()
    if db.empty:
        return None
    keywords = FOOD_KEYWORDS.get(food_class.lower(), [food_class.lower().replace("_", " ")])
    row = None
    for kw in keywords:
        mask = db["food"].str.contains(kw, case=False, na=False)
        if mask.any():
            row = db[mask].iloc[0]
            break
    if row is None:
        return None

    def safe(col, default=0):
        try:
            v = row[col]
            return round(float(v), 1) if pd.notna(v) else default
        except Exception:
            return default

    return {
        "calories": safe("Caloric Value"),
        "protein_g": safe("Protein"),
        "carbs_g": safe("Carbohydrates"),
        "fat_g": safe("Fat"),
        "fiber_g": safe("Dietary Fiber"),
        "sugar_g": safe("Sugars"),
        "vitamins": [c for c in VITAMIN_COLS if safe(c) > 0][:6],
        "minerals": [c for c in MINERAL_COLS if safe(c) > 0][:6],
        "serving_size": f"100g serving ({row['food']})",
        "source": "FOOD-DATA-GROUP CSV"
    }

# ── Generative Intelligence via new google-genai SDK ─────────────────────────
def get_nutrition(food: str) -> dict:
    result = lookup_nutrition(food)
    if result:
        return result

    prompt = f"""
    Give nutritional information for one serving of {food}.
    Return ONLY valid raw JSON matched to the following schema:
    {{
      "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0, "sugar_g": 0,
      "vitamins": ["Vitamin A"], "minerals": ["Iron"], "serving_size": "100 g"
    }}
    """
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def get_health_outcomes(food: str) -> dict:
    prompt = f"""
    Analyze the health effects of eating {food}.
    Return ONLY valid raw JSON matched to the following schema:
    {{
     "rating": 8, "benefits": ["", "", ""], "cautions": ["", ""], "good_for": ["", ""], "avoid_if": ["", ""], "summary": ""
    }}
    """
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def ask_vqa(question: str, food: str, history: list) -> str:
    conversation = "".join([f"{h['role']}: {h['content']}\n" for h in history[-6:]])
    prompt = f"You are NutriVision AI.\nFood: {food}\nPrevious Conversation:\n{conversation}\nUser Question:\n{question}\nAnswer clearly and briefly."
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text.strip()


# ── Render Hero Branding Header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🍎 NutriVision AI</h1>
  <p>Multimodal Nutrition Intelligence Platform · Upload a food image for instant nutritional insights</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar & Navigation Panel ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size:1.8rem;">🍎</span>
        <div style="font-size:1rem; font-weight:700; color:#10b981; margin-top:4px;">NutriVision AI Panel</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🚪 Log Out Dashboard", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
        
    st.markdown("### 🕓 Query History")
    if not st.session_state.history:
        st.markdown('<div style="color:#475569; font-size:0.85rem; text-align:center; padding: 1rem 0;">No logs created yet.</div>', unsafe_allow_html=True)
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - idx
            cal = entry["nutrition"].get("calories", "?") if entry["nutrition"] else "?"
            rating = entry["health"].get("rating", "?") if entry["health"] else "?"
            is_active = st.session_state.food_name == entry["food"]
            border_col = "#10b981" if is_active else "rgba(255,255,255,0.08)"
            bg_col = "rgba(16,185,129,0.08)" if is_active else "rgba(255,255,255,0.03)"

            st.markdown(f"""
            <div style="background:{bg_col}; border:1px solid {border_col}; border-radius:12px; padding:10px; margin-bottom:8px; display:flex; gap:10px; align-items:center;">
                <img src="data:image/jpeg;base64,{entry['thumb_b64']}" style="width:52px;height:40px;object-fit:cover;border-radius:7px;flex-shrink:0;"/>
                <div style="overflow:hidden;">
                    <div style="font-size:0.8rem;font-weight:600;color:#e2e8f0; white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{entry['food']}</div>
                    <div style="font-size:0.72rem;color:#10b981;margin-top:2px;">{cal} kcal &nbsp;·&nbsp; ⭐ {rating}/10</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Reload File Data", key=f"hist_{real_idx}", use_container_width=True):
                st.session_state.food_name = entry["food"]
                st.session_state.nutrition = entry["nutrition"]
                st.session_state.health = entry["health"]
                st.session_state.chat_history = entry["chat"]
                st.rerun()

        if st.button("🗑️ Clear History Logs", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ── Dynamic Workspace Layout Matrix ───────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-title">📸 Upload Food Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="Upload nutrition logs", label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        file_id = uploaded_file.file_id if hasattr(uploaded_file, "file_id") else uploaded_file.name
        if file_id != st.session_state.last_uploaded:
            st.session_state.last_uploaded = file_id
            st.session_state.food_name = None
            st.session_state.nutrition = None
            st.session_state.health = None
            st.session_state.chat_history = []
            st.session_state.confidence = None

            with st.spinner("🔍 Identifying food object with Gemini Matrix..."):
                try:
                    prompt = "Identify the food shown in this image. Return ONLY the simple food name string."
                    response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image])
                    predicted_label = response.text.strip().replace("```", "").replace("json", "").strip()
                    st.session_state.food_name = predicted_label
                    st.session_state.confidence = 100
                except Exception as e:
                    st.error(f"Gemini Processing Error: {e}")

        if st.session_state.food_name:
            conf = st.session_state.confidence
            conf_text = f"{conf}%" if conf is not None else "N/A"
            conf_color = "#10b981" if (conf or 0) >= 60 else "#f59e0b"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px 14px; margin-top:10px;">
                <div style="font-size:0.8rem;color:#94a3b8;">🤖 Model Prediction</div>
                <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:4px;">{st.session_state.food_name.replace('_',' ').title()}</div>
                <div style="font-size:0.8rem;color:{conf_color};margin-top:2px;">Confidence Score: {conf_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.markdown("**🔧 Manual Identity Adjustment Override:**")
        corrected = st.selectbox(
            "Select alternative food signature",
            options=TRAINED_CLASSES,
            index=TRAINED_CLASSES.index(st.session_state.food_name) if st.session_state.food_name in TRAINED_CLASSES else 0,
            label_visibility="collapsed"
        )
        if corrected != st.session_state.food_name:
            if st.button("✅ Apply Classification Adjustment"):
                st.session_state.food_name = corrected
                st.session_state.nutrition = None
                st.session_state.health = None
                st.session_state.chat_history = []
                st.rerun()

        if st.session_state.food_name and st.session_state.nutrition is None:
            with st.spinner("🥗 Assembling localized nutrition logs..."):
                try:
                    st.session_state.nutrition = get_nutrition(st.session_state.food_name)
                except Exception as e:
                    st.warning(f"Data lookup fallback caution: {e}")

        if st.session_state.food_name and st.session_state.health is None:
            with st.spinner("❤️ Parsing wellness outcome indexes..."):
                try:
                    st.session_state.health = get_health_outcomes(st.session_state.food_name)
                except Exception as e:
                    st.warning(f"Heuristics diagnostic timeout: {e}")

        if st.session_state.food_name and st.session_state.nutrition:
            thumb_buf = io.BytesIO()
            image.convert("RGB").resize((120, 90)).save(thumb_buf, format="JPEG", quality=70)
            thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()
            if st.session_state.food_name not in [e["food"] for e in st.session_state.history]:
                st.session_state.history.append({
                    "food": st.session_state.food_name, "nutrition": st.session_state.nutrition,
                    "health": st.session_state.health, "chat": [], "thumb_b64": thumb_b64
                })
                st.rerun()

with col_right:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
            <div style="font-size:4rem;">🍽️</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Upload a food image to get started</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; color:#334155;">Get nutritional facts, health insights, and ask questions</div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.food_name:
        food = st.session_state.food_name
        st.markdown(f'<div class="food-banner"><span>✅ Identified Food Signature</span><h2>🍽️ {food}</h2></div>', unsafe_allow_html=True)

        if st.session_state.nutrition:
            n = st.session_state.nutrition
            st.markdown('<div class="section-title">🥗 Nutritional Breakdown</div>', unsafe_allow_html=True)
            source_badge = '<span style="background:rgba(16,185,129,0.15); border-radius:20px; padding:2px 10px; font-size:0.72rem; color:#6ee7b7; margin-left:8px;">📊 From CSV Dataset</span>' if n.get("source") else '<span style="background:rgba(245,158,11,0.15); border-radius:20px; padding:2px 10px; font-size:0.72rem; color:#fcd34d; margin-left:8px;">🤖 AI Estimate Matrix</span>'
            st.markdown(f'<div style="color:#64748b; font-size:0.85rem; margin-bottom:0.8rem; display:flex; align-items:center;">Serving Context: {n.get("serving_size","")}{source_badge}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="nutr-card"><div class="value">{n.get("calories","—")}</div><div class="label">Calories</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="nutr-card"><div class="value">{n.get("protein_g","—")}g</div><div class="label">Protein</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="nutr-card"><div class="value">{n.get("carbs_g","—")}g</div><div class="label">Carbohydrates</div></div>', unsafe_allow_html=True)

            c4, c5, c6 = st.columns(3)
            c4.markdown(f'<div class="nutr-card"><div class="value">{n.get("fat_g","—")}g</div><div class="label">Total Fat</div></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="nutr-card"><div class="value">{n.get("fiber_g","—")}g</div><div class="label">Fiber</div></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="nutr-card"><div class="value">{n.get("sugar_g","—")}g</div><div class="label">Sugar</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            vc1, vc2 = st.columns(2)
            with vc1:
                if n.get("vitamins"):
                    st.markdown("**💊 Present Vitamins**")
                    st.markdown("".join(f'<span class="pill">{v}</span>' for v in n["vitamins"]), unsafe_allow_html=True)
            with vc2:
                if n.get("minerals"):
                    st.markdown("**🪨 Trace Minerals**")
                    st.markdown("".join(f'<span class="pill">{m}</span>' for m in n["minerals"]), unsafe_allow_html=True)

        if st.session_state.health:
            h = st.session_state.health
            st.markdown("<br><div class=\"section-title\">❤️ Comprehensive Wellness Index</div>", unsafe_allow_html=True)
            rating = h.get("rating", 5)
            color = "#10b981" if rating >= 7 else "#f59e0b" if rating >= 4 else "#ef4444"
            st.markdown(f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;"><span style="color:#94a3b8; font-size:0.9rem;">Health Compatibility Matrix</span><span style="font-size:1.2rem; font-weight:700; color:{color};">{rating}/10</span></div><div class="health-bar-wrap"><div class="health-bar-fill" style="width:{rating * 10}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color:#94a3b8; font-style:italic; font-size:0.9rem; margin-bottom:1rem;">{h.get("summary","")}</div>', unsafe_allow_html=True)

            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("**✅ Benefits**")
                for b in h.get("benefits", []):
                    st.markdown(f'<div class="benefit-item">🟢 {b}</div>', unsafe_allow_html=True)
            with hc2:
                st.markdown("**⚠️ Cautions & Risk Analysis**")
                for c in h.get("cautions", []):
                    st.markdown(f'<div class="benefit-item">🟡 {c}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            gc1, gc2 = st.columns(2)
            with gc1:
                if h.get("good_for"):
                    st.markdown("**👍 Recommended Target Profiles**")
                    st.markdown("".join(f'<span class="pill" style="background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3); color:#6ee7b7;">{g}</span>' for g in h["good_for"]), unsafe_allow_html=True)
            with gc2:
                if h.get("avoid_if"):
                    st.markdown("**🚫 Contraindications / Avoid If**")
                    st.markdown("".join(f'<span class="pill" style="background:rgba(239,68,68,0.1); border-color:rgba(239,68,68,0.3); color:#fca5a5;">{a}</span>' for a in h["avoid_if"]), unsafe_allow_html=True)

# ── VQA Interactivity Matrix ──────────────────────────────────────────────────
if st.session_state.food_name and st.session_state.food_name != "No food detected":
    st.markdown("---")
    st.markdown('<div class="section-title">💬 Ask About This Food</div>', unsafe_allow_html=True)
    
    for msg in st.session_state.chat_history:
        div_class = "chat-user" if msg["role"] == "user" else "chat-ai"
        icon = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{div_class}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    suggestions = [
        f"Is {st.session_state.food_name} good for continuous energy?",
        "Can a diabetic consume this processing signature?",
        "What is the breakdown of carbohydrates?",
        "How should I change my portion sizing for this?",
    ]
    st.markdown("**Quick Context Queries:**")
    q_cols = st.columns(2)
    for i, q in enumerate(suggestions):
        with q_cols[i % 2]:
            if st.button(q, key=f"sugg_{i}"):
                with st.spinner("🤖 Simulating answer logic..."):
                    answer = ask_vqa(q, st.session_state.food_name, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    with st.form("vqa_form", clear_on_submit=True):
        user_q = st.text_input("Ask a custom nutritional intelligence question...", placeholder="e.g. Can this lead to a glycemic spike?")
        if st.form_submit_button("Ask Platform →") and user_q.strip():
            with st.spinner("🤖 Querying Intelligence core..."):
                answer = ask_vqa(user_q, st.session_state.food_name, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()