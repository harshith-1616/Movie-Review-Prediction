import re
import string
import sys
from pathlib import Path

import joblib
import streamlit as st


# --- Preprocessing ----------------------------------------------------------
# The saved pipeline references this function by name (TfidfVectorizer's
# `preprocessor`), so it MUST be defined before joblib.load() is called.
def wordopt(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)        # URLs
    text = re.sub(r"<.*?>", "", text)                         # HTML tags
    text = re.sub(r"\[.*?\]", "", text)                       # [bracketed] text
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # punctuation
    text = re.sub(r"\w*\d\w*", "", text)                      # words with digits
    text = re.sub(r"\s+", " ", text).strip()                  # extra whitespace
    return text


# Make sure the unpickler can find wordopt however Streamlit runs this script.
sys.modules["__main__"].wordopt = wordopt

MODEL_CANDIDATES = [
    Path(__file__).parent / "Review.joblib",
    Path(r"C:\NLP\Sentimental\Review.joblib"),
]


@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "Review.joblib not found. Put it next to app.py or at "
        r"C:\NLP\Sentimental\Review.joblib"
    )


# --- Page setup & styling ---------------------------------------------------
st.set_page_config(page_title="CineScore | Movie Review Sentiment", page_icon="🎬", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&display=swap');

/* ---------- background: cinema spotlight + film grain ---------- */
.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 50% -5%, rgba(229, 9, 20, 0.35), transparent 70%),
        radial-gradient(ellipse 50% 35% at 90% 100%, rgba(245, 197, 24, 0.12), transparent 70%),
        radial-gradient(ellipse 50% 35% at 5% 90%, rgba(229, 9, 20, 0.12), transparent 70%),
        linear-gradient(180deg, #0b0b12 0%, #12121c 100%);
    background-attachment: fixed;
    color: #f2f2f2;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"], #MainMenu, footer { display: none; }
.block-container { padding-top: 0 !important; max-width: 820px; }

/* ---------- film strip border ---------- */
.filmstrip {
    height: 26px;
    margin: 0 -9999px;
    background:
        repeating-linear-gradient(90deg, transparent 0 14px, #0b0b12 14px 28px) center / 100% 12px no-repeat,
        #1a1a22;
    border-bottom: 2px solid #e50914;
}

/* ---------- navbar ---------- */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 4px; border-bottom: 1px solid rgba(255,255,255,0.08);
}
.logo {
    font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 3px;
    color: #e50914; text-shadow: 0 0 18px rgba(229, 9, 20, 0.55);
}
.logo span { color: #f5c518; }
.nav-links { display: flex; gap: 22px; font-size: 0.85rem; color: #b8b8c4; }
.nav-links b { color: #fff; font-weight: 600; }

/* ---------- hero ---------- */
.hero { text-align: center; padding: 38px 0 18px; }
.hero h1 {
    font-family: 'Bebas Neue', sans-serif; font-size: 4rem; line-height: 1;
    letter-spacing: 3px; margin: 0; color: #fff;
    text-shadow: 0 4px 30px rgba(229, 9, 20, 0.5);
}
.hero h1 em { color: #f5c518; font-style: normal; }
.hero p { color: #a9a9b8; font-size: 1.05rem; margin-top: 10px; }
.icons-row { font-size: 1.7rem; letter-spacing: 14px; margin-top: 14px; opacity: 0.9; }

/* ---------- cards ---------- */
.section-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.5rem; letter-spacing: 2px;
    color: #f5c518; margin: 26px 0 6px;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important; color: #fff !important;
    border: 1px solid rgba(255,255,255,0.18) !important; border-radius: 14px !important;
    font-size: 1rem !important; padding: 14px !important;
}
.stTextArea textarea:focus { border-color: #e50914 !important; box-shadow: 0 0 0 2px rgba(229,9,20,0.35) !important; }
.stTextArea label { display: none; }

/* buttons */
.stButton > button {
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.07); color: #eee; font-weight: 500;
    transition: all .2s ease;
}
.stButton > button:hover { border-color: #f5c518; color: #f5c518; transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e50914, #b20710); border: none; color: #fff;
    font-weight: 600; font-size: 1.05rem; letter-spacing: 1px; padding: 0.6rem 1rem;
    box-shadow: 0 6px 24px rgba(229, 9, 20, 0.45);
}
.stButton > button[kind="primary"]:hover { color: #fff; box-shadow: 0 8px 30px rgba(229, 9, 20, 0.7); }

/* ---------- result ticket ---------- */
.ticket {
    margin-top: 22px; padding: 26px; border-radius: 18px; text-align: center;
    border: 2px dashed rgba(255,255,255,0.25); position: relative;
    background: rgba(255,255,255,0.05); backdrop-filter: blur(6px);
}
.ticket.pos { border-color: #2ecc71; box-shadow: 0 0 40px rgba(46, 204, 113, 0.25); }
.ticket.neg { border-color: #e50914; box-shadow: 0 0 40px rgba(229, 9, 20, 0.30); }
.ticket .emoji { font-size: 4rem; }
.ticket .verdict { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; letter-spacing: 3px; }
.ticket.pos .verdict { color: #2ecc71; }
.ticket.neg .verdict { color: #ff4d57; }
.ticket .sub { color: #b8b8c4; margin-bottom: 14px; }
.stars { font-size: 1.6rem; color: #f5c518; letter-spacing: 4px; }
.bar { height: 12px; border-radius: 99px; background: rgba(255,255,255,0.12); overflow: hidden; margin: 14px 0 6px; }
.bar > div { height: 100%; border-radius: 99px; }
.pos .bar > div { background: linear-gradient(90deg, #1e8e50, #2ecc71); }
.neg .bar > div { background: linear-gradient(90deg, #b20710, #ff4d57); }
.bar-label { font-size: 0.85rem; color: #b8b8c4; }

.footer { text-align: center; color: #6c6c7a; font-size: 0.8rem; margin: 40px 0 18px; }
@media (max-width: 640px) { .hero h1 { font-size: 2.8rem; } .nav-links { display: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="filmstrip"></div>
    <div class="navbar">
        <div class="logo">CINE<span>SCORE</span> 🎬</div>
        <div class="nav-links"><b>Home</b><span>Reviews</span><span>Top Rated</span><span>About</span></div>
    </div>
    <div class="hero">
        <h1>What's the <em>Verdict?</em></h1>
        <p>Paste a movie review and let the AI critic tell you if it's a thumbs up or a thumbs down.</p>
        <div class="icons-row">🍿 🎥 🎞️ ⭐ 🎭</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# --- Sample reviews (click to fill the box) ---------------------------------
SAMPLES = {
    "🔥 Loved it": "An absolute masterpiece. The acting was brilliant and the story kept me hooked till the very end.",
    "💔 Hated it": "Terrible movie. The plot made no sense and the acting was painfully bad.",
    "😴 Boring": "I was bored within ten minutes. Slow, predictable, and a total waste of money.",
    "😂 Fun ride": "Hilarious and heartwarming. The whole family enjoyed it and we would happily watch it again.",
}


def fill(text):
    st.session_state["review"] = text


st.markdown('<div class="section-title">✍️ Write your review</div>', unsafe_allow_html=True)
review = st.text_area(
    "Your review", key="review", height=150,
    placeholder="Type or paste a movie review here...",
)

st.caption("🎟️ Or try a sample review:")
cols = st.columns(len(SAMPLES))
for col, (label, text) in zip(cols, SAMPLES.items()):
    col.button(label, on_click=fill, args=(text,), use_container_width=True)

go = st.button("🎬  ANALYZE SENTIMENT", type="primary", use_container_width=True)

# --- Prediction -------------------------------------------------------------
if go:
    if not review.strip():
        st.warning("🎟️ Please write or pick a review first.")
    elif not wordopt(review):
        st.warning("The review has no usable words after cleaning. Try a longer one.")
    else:
        try:
            pred = int(model.predict([review])[0])
            proba = model.predict_proba([review])[0]
            classes = list(model.classes_)
            p_pos = float(proba[classes.index(1)])
            confidence = p_pos if pred == 1 else 1 - p_pos
            filled = int(round(p_pos * 5))
            stars = "★" * filled + "☆" * (5 - filled)

            if pred == 1:
                css, emoji, verdict, sub = "pos", "👍", "THUMBS UP", "Audiences are going to love this one."
            else:
                css, emoji, verdict, sub = "neg", "👎", "THUMBS DOWN", "This one didn't impress the critic."

            st.markdown(
                f"""
                <div class="ticket {css}">
                    <div class="emoji">{emoji}</div>
                    <div class="verdict">{verdict}</div>
                    <div class="sub">{sub}</div>
                    <div class="stars">{stars}</div>
                    <div class="bar"><div style="width:{p_pos*100:.0f}%"></div></div>
                    <div class="bar-label">Positive score: {p_pos:.0%} &nbsp;•&nbsp; Model confidence: {confidence:.0%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except ValueError as e:
            # e.g. none of the words were in the training vocabulary
            st.warning(f"Couldn't score this review: {e}")

st.markdown('<div class="footer">🎞️ CineScore • Movie sentiment powered by TF-IDF + Logistic Regression</div>', unsafe_allow_html=True)
