import streamlit as st

# 1. Set the page to wide mode
st.set_page_config(layout="wide", page_title="nutritionAI Login")

# 2. Initialize the login state tracker
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 3. RENDER THE LOGIN SCREEN (If not logged in yet)
if not st.session_state.logged_in:
    # Injecting the exact color palette and layout styles from your layout design
    st.markdown("""
        <style>
            /* Create the deep dark teal radial background */
            .stApp {
                background: radial-gradient(circle at 50% 50%, #1a4243 0%, #10222a 60%, #091218 100%) !important;
            }
            
            /* Hide default Streamlit headers */
            [data-testid="stHeader"] {
                background: transparent !important;
            }
            
            /* CRITICAL FIX: Force the main viewport container to span full height and flex-center its contents */
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
            
            /* CRITICAL FIX: Force Streamlit's column block row wrapper to center itself */
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

            /* Ensure both columns share space properly inside the flex container */
            [data-testid="column"] {
                width: 50% !important;
                flex: 1 1 50% !important;
                max-width: 50% !important;
            }
            
            /* Custom styling for the User ID and Password boxes */
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
            
            /* Turning the standard Streamlit button into your premium pill button */
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

            /* Brand title style matching your layout */
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

    # Build the white container card layout split into columns
    col1, col2 = st.columns([1, 1.1])

    # Left Column: The Logo Artwork Panel
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

    # Right Column: The Input Forms Panel
    with col2:
        st.markdown('<h1 class="brand-title">nutrition<span>AI</span></h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter your workspace credentials to unlock tailored nutritional intelligence and analytics.</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Enter your name and enter the password.</p>', unsafe_allow_html=True)

        # Native interactive input widgets
        username = st.text_input("User ID", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••")

        # When clicked, this python button handles your redirect completely natively!
        if st.button("Sign In to Dashboard  →"):
            if username.strip() == "admin" and password.strip() == "nutrition2026":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid User ID or Password.")

    st.stop()

# ==========================================================================
# --- 4. YOUR ACTUAL MAIN PROJECT DASHBOARD CODE (Hidden until login) ---
# ==========================================================================
st.title("Welcome to nutritionAI Main Dashboard! 🎉")
st.sidebar.success("Logged in successfully as admin.")

if st.sidebar.button("Log Out"):
    st.session_state.logged_in = False
    st.rerun()

st.write("Your premium metrics, graphs, and main application system are safely unlocked.")


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
from groq import Groq
from dotenv import load_dotenv
from skimage.feature import hog

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriVision AI",
    page_icon="🍎",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Page background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hero header */
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

/* Section cards */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

/* Food result banner */
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

/* Nutrition card */
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

/* Vitamin / mineral pill */
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

/* Health rating bar */
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

/* Health outcome bullets */
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

/* Chat messages */
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

/* Section title */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* History bar */
.history-bar {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding: 1rem 0.5rem;
    margin-bottom: 1.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    scrollbar-width: thin;
    scrollbar-color: rgba(16,185,129,0.3) transparent;
}
.history-bar::-webkit-scrollbar { height: 4px; }
.history-bar::-webkit-scrollbar-track { background: transparent; }
.history-bar::-webkit-scrollbar-thumb { background: rgba(16,185,129,0.3); border-radius: 4px; }

.history-card {
    min-width: 110px;
    max-width: 110px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 8px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
    flex-shrink: 0;
}
.history-card:hover {
    border-color: rgba(16,185,129,0.4);
    background: rgba(16,185,129,0.06);
    transform: translateY(-2px);
}
.history-card.active {
    border-color: #10b981;
    background: rgba(16,185,129,0.1);
}
.history-card img {
    width: 90px;
    height: 70px;
    object-fit: cover;
    border-radius: 8px;
    margin-bottom: 6px;
}
.history-card .hc-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: #e2e8f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.history-card .hc-cal {
    font-size: 0.68rem;
    color: #10b981;
    margin-top: 2px;
}

/* Streamlit widget overrides */
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

# ── Groq client ────────────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

client = Groq(api_key=api_key)

# ── Session state init ─────────────────────────────────────────────────────────
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
    st.session_state.history = []   # list of dicts: {food, nutrition, health, chat, thumb_b64}
if "confidence" not in st.session_state:
    st.session_state.confidence = None


# ── Local model setup ──────────────────────────────────────────────────────────
IMG_SIZE = 128
MODEL_PATH = "models/food_classifier.pkl"

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

food_model = load_classifier()

# Trained classes (matches training dataset folders)
TRAINED_CLASSES = [
    "cake", "donuts", "french_fries", "fried_rice",
    "hot_dog", "ice_cream", "momos", "pizza", "waffles"
]


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract HOG + HSV color histogram features — identical to training."""
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    # HOG features
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # HSV color histograms
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features = color_features / (color_features.sum() + 1e-7)

    return np.concatenate([hog_features, color_features])


def predict_food(image: Image.Image):
    """Run local trained RandomForest classifier on a PIL image.
    Returns (label: str, confidence: float | None)
    """
    if food_model is None:
        return "Model not found", None

    # PIL (RGB) → numpy BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    features = extract_features(img_bgr).reshape(1, -1)

    label = food_model.predict(features)[0]

    # Confidence via predict_proba if available
    confidence = None
    if hasattr(food_model, "predict_proba"):
        proba = food_model.predict_proba(features)[0]
        confidence = round(float(proba.max()) * 100, 1)

    return label, confidence


# ── Nutrition database from CSV ──────────────────────────────────────────────────────
VITAMIN_COLS  = ["Vitamin A","Vitamin B1","Vitamin B2","Vitamin B3",
                 "Vitamin B5","Vitamin B6","Vitamin B11","Vitamin B12",
                 "Vitamin C","Vitamin D","Vitamin E","Vitamin K"]
MINERAL_COLS  = ["Calcium","Iron","Magnesium","Potassium",
                 "Zinc","Phosphorus","Copper","Manganese","Selenium"]

# Keyword map: trained class -> search terms for CSV
FOOD_KEYWORDS = {
    "waffles":      ["waffle"],
    "pizza":        ["pizza"],
    "hot_dog":      ["hotdog", "hot dog"],
    "french_fries": ["french fries", "french frie"],
    "fried_rice":   ["fried rice"],
    "ice_cream":    ["ice cream", "gelato", "sundae"],
    "cake":         ["cake"],
    "donuts":       ["donut", "doughnut"],
    "momos":        ["momos", "dumpling"],
    "noodles":      ["noodle"],
    "omelette":     ["omelette", "omelet", "egg"],
    "samosa":       ["samosa"],
}

@st.cache_data
def load_nutrition_db() -> pd.DataFrame:
    """Merge all 5 CSVs into one DataFrame."""
    files = sorted(glob.glob("data/FOOD-DATA-GROUP*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, encoding="latin1"))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def lookup_nutrition(food_class: str) -> dict | None:
    """Look up nutrition from CSV for a trained food class label.
    Returns a dict matching the UI schema, or None if not found.
    """
    db = load_nutrition_db()
    if db.empty:
        return None

    keywords = FOOD_KEYWORDS.get(food_class.lower(), [food_class.lower().replace("_", " ")])

    row = None
    for kw in keywords:
        mask = db["food"].str.contains(kw, case=False, na=False)
        if mask.any():
            row = db[mask].iloc[0]   # take first match
            break

    if row is None:
        return None

    def safe(col, default=0):
        try:
            v = row[col]
            return round(float(v), 1) if pd.notna(v) else default
        except Exception:
            return default

    # Which vitamins are present (value > 0)?
    vitamins = [c for c in VITAMIN_COLS if safe(c) > 0]
    minerals = [c for c in MINERAL_COLS if safe(c) > 0]

    return {
        "calories":    safe("Caloric Value"),
        "protein_g":   safe("Protein"),
        "carbs_g":     safe("Carbohydrates"),
        "fat_g":       safe("Fat"),
        "fiber_g":     safe("Dietary Fiber"),
        "sugar_g":     safe("Sugars"),
        "vitamins":    vitamins[:6],    # show up to 6
        "minerals":    minerals[:6],
        "serving_size": f"100g serving ({row['food']})",
        "source":      "FOOD-DATA-GROUP CSV"
    }


def get_nutrition(food: str) -> dict:
    """Get nutrition from CSV first; fall back to Groq only if not found."""
    result = lookup_nutrition(food)
    if result:
        return result

    # Fallback: Groq text (only if CSV has no match)
    prompt = f"""Give nutritional info for one typical serving of "{food}".
Return ONLY a JSON object:
{{
  "calories": <number>, "protein_g": <number>, "carbs_g": <number>,
  "fat_g": <number>, "fiber_g": <number>, "sugar_g": <number>,
  "vitamins": ["Vitamin A"], "minerals": ["Iron"],
  "serving_size": "<e.g. 1 waffle (75g)>"
}}
Return raw JSON only."""
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250, temperature=0.1
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)



def get_health_outcomes(food: str) -> dict:
    prompt = f"""Analyze the health outcomes of eating "{food}".
Return ONLY a JSON object with exactly these keys:
{{
  "rating": <integer 1-10>,
  "benefits": ["benefit 1", "benefit 2", "benefit 3"],
  "cautions": ["caution 1", "caution 2"],
  "good_for": ["Athletes", "Children"],
  "avoid_if": ["Diabetics", "Lactose intolerant"],
  "summary": "<one sentence health summary>"
}}
Return raw JSON only, no markdown, no explanation."""

    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.1
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def ask_vqa(question: str, food: str, history: list) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                f"You are NutriVision AI, a nutrition expert. "
                f"The user has uploaded an image of '{food}'. "
                f"Answer all questions about its nutrition, health impacts, and dietary advice. "
                f"Be concise, friendly, and accurate."
            )
        }
    ]
    for h in history[-6:]:  # keep last 6 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        max_tokens=300, temperature=0.5
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🍎 NutriVision AI</h1>
  <p>Multimodal Nutrition Intelligence Platform · Upload a food image for instant nutritional insights</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar History Panel ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size:1.8rem;">🍎</span>
        <div style="font-size:1rem; font-weight:700; color:#10b981; margin-top:4px;">NutriVision AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🕓 History")

    if not st.session_state.history:
        st.markdown("""
        <div style="color:#475569; font-size:0.85rem; text-align:center; padding: 1rem 0;">
            No history yet.<br>Upload a food image to start.
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - idx
            cal  = entry["nutrition"].get("calories", "?") if entry["nutrition"] else "?"
            rating = entry["health"].get("rating", "?") if entry["health"] else "?"
            is_active = st.session_state.food_name == entry["food"]
            border_color = "#10b981" if is_active else "rgba(255,255,255,0.08)"
            bg_color = "rgba(16,185,129,0.08)" if is_active else "rgba(255,255,255,0.03)"

            st.markdown(f"""
            <div style="
                background:{bg_color};
                border:1px solid {border_color};
                border-radius:12px;
                padding:10px;
                margin-bottom:8px;
                display:flex;
                gap:10px;
                align-items:center;
            ">
                <img src="data:image/jpeg;base64,{entry['thumb_b64']}"
                     style="width:52px;height:40px;object-fit:cover;border-radius:7px;flex-shrink:0;"/>
                <div style="overflow:hidden;">
                    <div style="font-size:0.8rem;font-weight:600;color:#e2e8f0;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {entry['food']}
                    </div>
                    <div style="font-size:0.72rem;color:#10b981;margin-top:2px;">
                        {cal} kcal &nbsp;·&nbsp; ⭐ {rating}/10
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Load", key=f"hist_{real_idx}", width="stretch"):
                st.session_state.food_name    = entry["food"]
                st.session_state.nutrition    = entry["nutrition"]
                st.session_state.health       = entry["health"]
                st.session_state.chat_history = entry["chat"]
                st.rerun()

        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()


# ── Layout: left = upload, right = results ─────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="section-title">📸 Upload Food Image</div>', unsafe_allow_html=True)
    # UPDATE LINE 774 TO THIS:
    uploaded_file = st.file_uploader(
        label="Upload nutrition logs", 
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="", width="stretch")

        file_id = uploaded_file.file_id if hasattr(uploaded_file, "file_id") else uploaded_file.name
        if file_id != st.session_state.last_uploaded:
            st.session_state.last_uploaded = file_id
            st.session_state.food_name = None
            st.session_state.nutrition = None
            st.session_state.health = None
            st.session_state.chat_history = []
            st.session_state.confidence = None

            with st.spinner("🔍 Classifying with trained model..."):
                try:
                    predicted_label, conf = predict_food(image)
                    st.session_state.food_name = predicted_label
                    st.session_state.confidence = conf
                except Exception as e:
                    st.error(f"Classification error: {e}")

        # ── Confidence display ─────────────────────────────────────────────
        if st.session_state.food_name:
            conf = st.session_state.confidence
            conf_text = f"{conf}%" if conf is not None else "N/A"
            conf_color = "#10b981" if (conf or 0) >= 60 else "#f59e0b" if (conf or 0) >= 35 else "#ef4444"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                        border-radius:10px;padding:10px 14px;margin-top:10px;">
                <div style="font-size:0.8rem;color:#94a3b8;">🤖 Model Prediction</div>
                <div style="font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:4px;">
                    {st.session_state.food_name.replace('_',' ').title()}
                </div>
                <div style="font-size:0.8rem;color:{conf_color};margin-top:2px;">
                    Confidence: {conf_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Correction dropdown ────────────────────────────────────────────
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.markdown("**🔧 Correct if wrong:**")
        corrected = st.selectbox(
            "Select correct food class",
            options=TRAINED_CLASSES,
            index=TRAINED_CLASSES.index(st.session_state.food_name)
                  if st.session_state.food_name in TRAINED_CLASSES else 0,
            key="correction_select",
            label_visibility="collapsed"
        )
        if corrected != st.session_state.food_name:
            if st.button("✅ Apply Correction", key="apply_correction"):
                st.session_state.food_name = corrected
                st.session_state.nutrition = None
                st.session_state.health = None
                st.session_state.chat_history = []

        # ── Fetch nutrition + health once food is confirmed ────────────────
        if st.session_state.food_name and st.session_state.nutrition is None:
            with st.spinner("🥗 Fetching nutrition data..."):
                try:
                    st.session_state.nutrition = get_nutrition(st.session_state.food_name)
                except Exception as e:
                    st.warning(f"Nutrition fetch error: {e}")

        if st.session_state.food_name and st.session_state.health is None:
            with st.spinner("❤️ Analyzing health outcomes..."):
                try:
                    st.session_state.health = get_health_outcomes(st.session_state.food_name)
                except Exception as e:
                    st.warning(f"Health analysis error: {e}")

        # ── Save to history ────────────────────────────────────────────────
        if st.session_state.food_name and st.session_state.nutrition:
            thumb_buf = io.BytesIO()
            image.convert("RGB").resize((120, 90)).save(thumb_buf, format="JPEG", quality=70)
            thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode()

            existing = [e["food"] for e in st.session_state.history]
            if st.session_state.food_name not in existing:
                st.session_state.history.append({
                    "food":      st.session_state.food_name,
                    "nutrition": st.session_state.nutrition,
                    "health":    st.session_state.health,
                    "chat":      [],
                    "thumb_b64": thumb_b64
                })
                st.rerun()

# ── Right column: results ──────────────────────────────────────────────────────
with col_right:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
            <div style="font-size:4rem;">🍽️</div>
            <div style="font-size:1.1rem; margin-top:1rem;">Upload a food image to get started</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; color:#334155;">
                Get nutritional facts, health insights, and ask questions
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.food_name:
        food = st.session_state.food_name

        # Food identified banner
        st.markdown(f"""
        <div class="food-banner">
            <span>✅ Identified as</span>
            <h2>🍽️ {food}</h2>
        </div>
        """, unsafe_allow_html=True)

        # ── Nutritional Breakdown ──────────────────────────────────────────────
        if st.session_state.nutrition:
            n = st.session_state.nutrition
            st.markdown('<div class="section-title">🥗 Nutritional Breakdown</div>', unsafe_allow_html=True)
            source = n.get("source", "")
            source_badge = (
                '<span style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);'
                'border-radius:20px;padding:2px 10px;font-size:0.72rem;color:#6ee7b7;margin-left:8px;">'
                '📊 From CSV Dataset</span>'
            ) if source else (
                '<span style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);'
                'border-radius:20px;padding:2px 10px;font-size:0.72rem;color:#fcd34d;margin-left:8px;">'
                '🤖 AI Estimate</span>'
            )
            st.markdown(
                f'<div style="color:#64748b;font-size:0.85rem;margin-bottom:0.8rem;display:flex;align-items:center;">'
                f'Per serving: {n.get("serving_size","")}{source_badge}</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("calories","—")}</div><div class="label">Calories</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("protein_g","—")}g</div><div class="label">Protein</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("carbs_g","—")}g</div><div class="label">Carbs</div></div>', unsafe_allow_html=True)

            c4, c5, c6 = st.columns(3)
            with c4:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("fat_g","—")}g</div><div class="label">Fat</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("fiber_g","—")}g</div><div class="label">Fiber</div></div>', unsafe_allow_html=True)
            with c6:
                st.markdown(f'<div class="nutr-card"><div class="value">{n.get("sugar_g","—")}g</div><div class="label">Sugar</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            vc1, vc2 = st.columns(2)
            with vc1:
                vitamins = n.get("vitamins", [])
                if vitamins:
                    st.markdown("**💊 Vitamins**")
                    pills = "".join(f'<span class="pill">{v}</span>' for v in vitamins)
                    st.markdown(pills, unsafe_allow_html=True)
            with vc2:
                minerals = n.get("minerals", [])
                if minerals:
                    st.markdown("**🪨 Minerals**")
                    pills = "".join(f'<span class="pill">{m}</span>' for m in minerals)
                    st.markdown(pills, unsafe_allow_html=True)

        # ── Health Outcomes ────────────────────────────────────────────────────
        if st.session_state.health:
            h = st.session_state.health
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">❤️ Health Outcomes</div>', unsafe_allow_html=True)

            rating = h.get("rating", 5)
            bar_width = rating * 10
            color = "#10b981" if rating >= 7 else "#f59e0b" if rating >= 4 else "#ef4444"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
                <span style="color:#94a3b8; font-size:0.9rem;">Health Rating</span>
                <span style="font-size:1.2rem; font-weight:700; color:{color};">{rating}/10</span>
            </div>
            <div class="health-bar-wrap">
                <div class="health-bar-fill" style="width:{bar_width}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div style="color:#94a3b8; font-style:italic; font-size:0.9rem; margin-bottom:1rem;">{h.get("summary","")}</div>', unsafe_allow_html=True)

            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("**✅ Benefits**")
                for b in h.get("benefits", []):
                    st.markdown(f'<div class="benefit-item">🟢 {b}</div>', unsafe_allow_html=True)
            with hc2:
                st.markdown("**⚠️ Cautions**")
                for c in h.get("cautions", []):
                    st.markdown(f'<div class="benefit-item">🟡 {c}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            gc1, gc2 = st.columns(2)
            with gc1:
                good = h.get("good_for", [])
                if good:
                    st.markdown("**👍 Good for**")
                    pills = "".join(f'<span class="pill" style="background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3); color:#6ee7b7;">{g}</span>' for g in good)
                    st.markdown(pills, unsafe_allow_html=True)
            with gc2:
                avoid = h.get("avoid_if", [])
                if avoid:
                    st.markdown("**🚫 Avoid if**")
                    pills = "".join(f'<span class="pill" style="background:rgba(239,68,68,0.1); border-color:rgba(239,68,68,0.3); color:#fca5a5;">{a}</span>' for a in avoid)
                    st.markdown(pills, unsafe_allow_html=True)

# ── VQA Chat Section ───────────────────────────────────────────────────────────
if st.session_state.food_name and st.session_state.food_name != "No food detected":
    st.markdown("---")
    st.markdown('<div class="section-title">💬 Ask About This Food</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#64748b; font-size:0.85rem; margin-bottom:1rem;">Ask anything about <strong style="color:#10b981;">{st.session_state.food_name}</strong> — nutrition, health, diet tips, recipes...</div>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Suggested questions
    suggestions = [
        f"Is {st.session_state.food_name} good for weight loss?",
        "Can a diabetic eat this?",
        "What is the calorie count?",
        "What vitamins does this have?",
    ]
    st.markdown("**Quick questions:**")
    q_cols = st.columns(2)
    for i, q in enumerate(suggestions):
        with q_cols[i % 2]:
            if st.button(q, key=f"sugg_{i}"):
                with st.spinner("🤖 Thinking..."):
                    answer = ask_vqa(q, st.session_state.food_name, st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    # Custom question input
    with st.form("vqa_form", clear_on_submit=True):
        user_q = st.text_input("Or type your own question...", placeholder="e.g. How much protein per serving?")
        submitted = st.form_submit_button("Ask →")
        if submitted and user_q.strip():
            with st.spinner("🤖 Thinking..."):
                answer = ask_vqa(user_q, st.session_state.food_name, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})