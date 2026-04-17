import os
import base64
import requests
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StyleMatch · Similar Item Finder",
    page_icon="✦",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0e0c0a !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a1410 0%, #0e0c0a 60%) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    border-bottom: 1px solid rgba(212,180,130,0.15);
    margin-bottom: 2.5rem;
}

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #c9a96e;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    font-weight: 300;
}

.hero-title span {
    color: #c9a96e;
    font-style: italic;
}

.hero-sub {
    margin-top: 1rem;
    font-size: 0.88rem;
    color: #7a7269;
}

.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #c9a96e;
    margin-bottom: 1.2rem;
}

.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(201,169,110,0.35), transparent);
    margin: 2rem 0;
}

.amazon-btn {
    display: block;
    text-align: center;
    margin-top: 6px;
    padding: 5px 8px;
    background: rgba(201,169,110,0.12);
    border: 1px solid rgba(201,169,110,0.3);
    border-radius: 6px;
    color: #c9a96e !important;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-decoration: none !important;
    transition: background 0.2s;
}

.amazon-btn:hover {
    background: rgba(201,169,110,0.25);
}

.match-label {
    text-align: center;
    font-size: 0.7rem;
    color: #7a7269;
    margin-top: 4px;
}

.source-header {
    font-size: 0.68rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #c9a96e;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(201,169,110,0.15);
}

.query-pill {
    display: inline-block;
    background: rgba(201,169,110,0.1);
    border: 1px solid rgba(201,169,110,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.72rem;
    color: #c9a96e;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ AI-Powered Visual Search ✦</div>
    <div class="hero-title">Style<span>Match</span></div>
    <div class="hero-sub">Upload clothing — discover visually similar pieces from your collection & Amazon</div>
</div>
""", unsafe_allow_html=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
SERPAPI_KEY = "92975fcfa35f32d1d0de7be9ddeadaa446835284b8edacc5c271d4978bb3c686"   # https://serpapi.com
IMGBB_KEY   = "8905c15f51970d8bb2e31c4570b6ccb6"     # https://api.imgbb.com

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

@st.cache_data
def load_features():
    return np.load("features/features.npy"), np.load("features/names.npy")

model = load_model()
feature_list, image_names = load_features()

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.expand_dims(np.array(img), axis=0)
    arr = preprocess_input(arr)
    return model.predict(arr, verbose=0).flatten()

# ── Dataset recommendations (top 3) ──────────────────────────────────────────
def recommend_from_dataset(features):
    similarity = cosine_similarity([features], feature_list)[0]
    indices = np.argsort(similarity)[::-1][1:4]
    return indices, similarity

# ── PIL image to base64 ───────────────────────────────────────────────────────
def pil_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ── Upload to imgbb → public URL ──────────────────────────────────────────────
def upload_to_imgbb(img: Image.Image) -> str:
    b64 = pil_to_base64(img)
    resp = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_KEY, "image": b64},
        timeout=15,
    )
    data = resp.json()
    if data.get("success"):
        return data["data"]["url"]
    else:
        raise Exception(f"imgbb upload failed: {data}")

# ── Use Claude vision to extract style tags from image ────────────────────────
def describe_style_with_claude(img: Image.Image) -> str:
    """
    Sends image to Claude and asks for a short shopping-style description
    of the garment — style, pattern, cut, fabric feel — ignoring color.
    We ignore color because the user's keyword will override it.
    """
    try:
        b64 = pil_to_base64(img)
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 60,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this clothing item for a shopping search query. "
                                "Focus on: garment type, pattern, cut, neckline, sleeve style, fabric feel. "
                                "Do NOT mention color. "
                                "Reply with only 4-6 words, no punctuation. "
                                "Example: floral maxi dress ruffle hem"
                            ),
                        },
                    ],
                }
            ],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        data = resp.json()
        style_tags = data["content"][0]["text"].strip().lower()
        return style_tags
    except Exception:
        return "clothing apparel"

# ── Build merged query: keyword + image style ─────────────────────────────────
def build_combined_query(keyword: str, img: Image.Image) -> str:
    style = describe_style_with_claude(img)
    # keyword goes first (user intent), style tags follow (visual context)
    combined = f"{keyword.strip()} {style}"
    return combined

# ── Google Shopping search (India, INR) ──────────────────────────────────────
def search_by_keyword(query: str, n: int = 3):
    try:
        params = {
            "engine"       : "google_shopping",
            "q"            : query,
            "api_key"      : SERPAPI_KEY,
            "num"          : 10,
            "hl"           : "en",
            "gl"           : "in",
            "google_domain": "google.co.in",
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        data = resp.json()

        results = []
        for item in data.get("shopping_results", []):
            link  = item.get("link") or item.get("product_link") or ""
            image = item.get("thumbnail") or ""
            title = item.get("title", "Product")
            price = item.get("price", "")
            if image and link:
                results.append({"title": title, "image": image, "link": link, "price": price})
            if len(results) == n:
                break
        return results

    except Exception as e:
        st.error(f"Search failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

# ── Google Lens visual search (image only, no keyword) ────────────────────────
def search_by_image(img: Image.Image, n: int = 3):
    try:
        image_url = upload_to_imgbb(img)
        params = {
            "engine" : "google_lens",
            "url"    : image_url,
            "api_key": SERPAPI_KEY,
            "hl"     : "en",
            "country": "in",
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=20)
        data = resp.json()

        results = []
        for item in data.get("visual_matches", []):
            link  = item.get("link", "")
            image = item.get("thumbnail") or item.get("image", "")
            title = item.get("title", "Product")
            price = item.get("price", {})
            price_text = price.get("value", "") if isinstance(price, dict) else str(price)
            if image and link:
                results.append({"title": title, "image": image, "link": link, "price": price_text})
            if len(results) == n:
                break
        return results

    except Exception as e:
        st.error(f"Image search failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return []

# ── Render product cards ──────────────────────────────────────────────────────
def render_products(results):
    cols = st.columns(3)
    for col, product in zip(cols, results):
        with col:
            st.image(product["image"], use_container_width=True)
            price_text = product["price"] if product["price"] else ""
            st.markdown(f"""
            <div class="match-label" style="margin-bottom:4px;">
                {product['title'][:40]}{'...' if len(product['title']) > 40 else ''}
                {'<br>' + price_text if price_text else ''}
            </div>
            <a class="amazon-btn" href="{product['link']}" target="_blank">
                View Product ↗
            </a>
            """, unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Search</div>', unsafe_allow_html=True)

    user_query = st.text_input(
        "Keyword (optional)",
        placeholder="e.g. red dress, black kurta...",
        help="Leave blank for pure visual search. Add a keyword to override color/type while keeping the style from your image."
    )

    uploaded_file = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Your Item", use_container_width=True)

with right:

    has_image   = uploaded_file is not None
    has_keyword = user_query.strip() != ""

    # ── Case 1: image + keyword → Claude merges style + keyword ──────────────
    if has_image and has_keyword:
        with st.spinner("Analyzing..."):
            features = extract_features(img)
            indices, similarity = recommend_from_dataset(features)

        avg_score = np.mean([similarity[i] for i in indices])
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(201,169,110,0.2);
        border-radius:12px;padding:1rem;margin-bottom:1rem;font-size:0.8rem;color:#c9a96e;">
        <b>Model Performance</b><br>Avg Similarity: {round(avg_score * 100, 2)}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="source-header">From Your Collection</div>', unsafe_allow_html=True)
        dataset_cols = st.columns(3)
        for rank, (col, i) in enumerate(zip(dataset_cols, indices), start=1):
            with col:
                st.image(f"dataset/{image_names[i]}", use_container_width=True)
                st.markdown(f'<div class="match-label">MATCH {rank} · {round(similarity[i]*100,2)}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="source-header">Shop Similar on Amazon</div>', unsafe_allow_html=True)
        with st.spinner("Reading image style + applying your keyword..."):
            combined_query = build_combined_query(user_query.strip(), img)

        st.markdown(f'<div class="query-pill">Searching for: {combined_query}</div>', unsafe_allow_html=True)

        with st.spinner("Finding products..."):
            results = search_by_keyword(combined_query, n=3)

        if results:
            render_products(results)
        else:
            st.markdown('<div style="color:#7a7269;font-size:0.85rem;">No results found. Try a different keyword.</div>', unsafe_allow_html=True)

    # ── Case 2: image only → Google Lens visual search ────────────────────────
    elif has_image and not has_keyword:
        with st.spinner("Analyzing..."):
            features = extract_features(img)
            indices, similarity = recommend_from_dataset(features)

        avg_score = np.mean([similarity[i] for i in indices])
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(201,169,110,0.2);
        border-radius:12px;padding:1rem;margin-bottom:1rem;font-size:0.8rem;color:#c9a96e;">
        <b>Model Performance</b><br>Avg Similarity: {round(avg_score * 100, 2)}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="source-header">From Your Collection</div>', unsafe_allow_html=True)
        dataset_cols = st.columns(3)
        for rank, (col, i) in enumerate(zip(dataset_cols, indices), start=1):
            with col:
                st.image(f"dataset/{image_names[i]}", use_container_width=True)
                st.markdown(f'<div class="match-label">MATCH {rank} · {round(similarity[i]*100,2)}%</div>', unsafe_allow_html=True)

        st.markdown('<div class="source-header">Shop Similar on Amazon</div>', unsafe_allow_html=True)
        with st.spinner("Finding visually similar products..."):
            results = search_by_image(img, n=3)

        if results:
            render_products(results)
        else:
            st.markdown('<div style="color:#7a7269;font-size:0.85rem;">Could not fetch results. Check your API keys.</div>', unsafe_allow_html=True)

    # ── Case 3: keyword only → Google Shopping ────────────────────────────────
    elif has_keyword and not has_image:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="source-header">Shop Similar on Amazon</div>', unsafe_allow_html=True)
        with st.spinner("Searching by keyword..."):
            results = search_by_keyword(user_query.strip(), n=3)

        if results:
            render_products(results)
        else:
            st.markdown('<div style="color:#7a7269;font-size:0.85rem;">No results found. Try a different keyword.</div>', unsafe_allow_html=True)

    # ── Case 4: nothing yet ───────────────────────────────────────────────────
    else:
        st.info("Type a keyword, upload an image, or both.")