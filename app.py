import json
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

# Streamlit Page Config
st.set_page_config(
    page_title="Multimodal Search System (CLIP + FAISS)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .metric-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .result-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.2);
        border-color: #38bdf8;
    }
    
    .similarity-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-high {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-mid {
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .badge-low {
        background: rgba(234, 179, 8, 0.2);
        color: #fde047;
        border: 1px solid rgba(234, 179, 8, 0.4);
    }
</style>
""", unsafe_allow_html=True)

from src.config import FAISS_INDEX_PATH, INDEX_METADATA_PATH, EMBEDDINGS_NPY_PATH, EVAL_REPORT_PATH, EVAL_PLOT_PATH, FAILURE_ANALYSIS_PATH, DEVICE
from src.indexer import FAISSVectorIndexer
from src.query_agent import RetrievalAgent, format_structured_result
from src.visualization import project_embeddings_2d, plot_embedding_space, plot_query_neighborhood_2d

@st.cache_resource(show_spinner=False)
def get_cached_indexer():
    indexer = FAISSVectorIndexer()
    if not indexer.load_index():
        st.error("FAISS index not found! Please run `python build_index.py` first.")
        st.stop()
    return indexer

indexer = get_cached_indexer()
retrieval_agent = RetrievalAgent(indexer)

# Sidebar Controls
st.sidebar.header("⚙️ Search & Filter Controls")

if st.sidebar.button("🔄 Force Refresh Index"):
    st.cache_resource.clear()
    st.rerun()

top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=24, value=8, step=1)
similarity_threshold = st.sidebar.slider("Min Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

# Facet Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Facet Filters")
available_categories = ["All"] + sorted(indexer.metadata_df["category"].unique().tolist())
selected_category = st.sidebar.selectbox("Filter by Category", available_categories)

available_colors = ["All"] + sorted(indexer.metadata_df["color"].unique().tolist())
selected_color = st.sidebar.selectbox("Filter by Color", available_colors)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Hardware Device:** `{DEVICE}`")
st.sidebar.markdown(f"**Indexed Vectors:** `{indexer.index.ntotal}`")
st.sidebar.markdown(f"**Vector Dimension:** `512-D`")

# Header Card
st.markdown("""
<div class="header-card">
    <div class="header-title">⚡ Multimodal Search System</div>
    <div style="color: #cbd5e1; font-size: 1.05rem;">
        Cross-Modal Retrieval powered by <b>OpenAI CLIP (512-D)</b>, <b>FAISS Vector Indexing</b>, and <b>Vector Latent Arithmetic</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Text → Image Search",
    "🖼️ Image → Image / Text",
    "⚖️ Hybrid Multimodal Search",
    "➕ Ingest Image to Index",
    "📊 Evaluation & Benchmarks",
    "🗃️ Embedding Space Explorer",
    "🧭 Structured Query Agent"
])

# ---------------------------------------------------------
# TAB 1: TEXT TO IMAGE SEARCH WITH NEGATIVE PROMPTING
# ---------------------------------------------------------
with tab1:
    st.subheader("Open-Vocabulary Text-to-Image Search")
    st.markdown("Type **ANY free-text description** below to query the 512-D CLIP embedding space in real time.")

    # Preset Suggestions
    st.markdown("**Quick Preset Queries:**")
    q_cols = st.columns(6)
    if q_cols[0].button("🏎️ Red sports car"):
        st.session_state["search_input"] = "red sports car"
    if q_cols[1].button("🚲 Mountain bike"):
        st.session_state["search_input"] = "mountain bike"
    if q_cols[2].button("🎧 Headphones"):
        st.session_state["search_input"] = "black wireless headphones"
    if q_cols[3].button("🐕 Golden retriever"):
        st.session_state["search_input"] = "happy golden retriever dog"
    if q_cols[4].button("☕ Espresso coffee"):
        st.session_state["search_input"] = "steaming cup of espresso coffee"
    if q_cols[5].button("🍕 Gourmet pizza"):
        st.session_state["search_input"] = "fresh baked pepperoni pizza"

    if "search_input" not in st.session_state:
        st.session_state["search_input"] = "red sports car"

    col_q1, col_q2 = st.columns([2, 1])
    with col_q1:
        user_query = st.text_input(
            "Enter positive search prompt:",
            key="search_input",
            placeholder="e.g. 'sports car', 'running shoes', 'motorcycle'..."
        )

    with col_q2:
        negative_prompt = st.text_input(
            "➖ Negative prompt (exclude attributes):",
            value="",
            placeholder="e.g. 'vintage', 'red', 'boots'..."
        )

    beta = 0.4
    if negative_prompt.strip():
        beta = st.slider("Negative Penalty Weight (β):", min_value=0.1, max_value=1.0, value=0.45, step=0.05,
                         help="v_final = normalize(v_pos - β * v_neg). Higher β suppresses unwanted visual features more aggressively.")

    if user_query.strip():
        fetch_k = top_k * 4 if (selected_category != "All" or selected_color != "All") else top_k

        # Check if negative prompting is active
        if negative_prompt.strip():
            raw_results, latency_ms = indexer.search_with_negative_prompt(user_query, negative_prompt, beta=beta, top_k=fetch_k)
            st.info(f"✨ **Vector Arithmetic Active:** `normalize(v('{user_query}') - {beta} × v('{negative_prompt}'))`")
        else:
            raw_results, latency_ms = indexer.search_by_text(user_query, top_k=fetch_k)

        # Apply facet filters
        filtered_results = []
        for r in raw_results:
            if selected_category != "All" and r["category"] != selected_category:
                continue
            if selected_color != "All" and r["color"] != selected_color:
                continue
            if r["similarity_score"] >= similarity_threshold:
                filtered_results.append(r)
            if len(filtered_results) >= top_k:
                break

        st.markdown(f"⏱️ **Search Latency:** `{latency_ms:.2f} ms` | Query: **'{user_query}'** | Showing **{len(filtered_results)}** matches")
        if selected_category != "All" or selected_color != "All":
            st.caption(f"Filters Active — Category: `{selected_category}`, Color: `{selected_color}`")
        st.markdown("---")

        if not filtered_results:
            st.warning("No images matched active filters or threshold.")
        else:
            cols_per_row = 4
            for i in range(0, len(filtered_results), cols_per_row):
                row_items = filtered_results[i:i+cols_per_row]
                grid_cols = st.columns(cols_per_row)
                for idx, item in enumerate(row_items):
                    with grid_cols[idx]:
                        score = item["similarity_score"]
                        badge_class = "badge-high" if score >= 0.25 else ("badge-mid" if score >= 0.20 else "badge-low")
                        
                        st.image(item["image_path"], use_container_width=True)
                        st.markdown(f"**Rank #{i + idx + 1}**")
                        st.markdown(f"<span class='similarity-badge {badge_class}'>Cosine Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                        st.caption(f"**Caption:** {item['caption']}")
                        st.caption(f"🏷️ `{item['category']}` | 🎨 `{item['color']}`")

# ---------------------------------------------------------
# TAB 2: IMAGE TO IMAGE / TEXT SEARCH
# ---------------------------------------------------------
with tab2:
    st.subheader("Image-to-Image & Image-to-Text Search")
    st.markdown("Upload any image or choose a dataset image to discover nearest visual and semantic neighbors.")

    input_mode = st.radio("Select Image Source:", ["Pick Sample Dataset Image", "Upload Custom Image"], horizontal=True)
    query_img = None

    if input_mode == "Pick Sample Dataset Image":
        sample_paths = indexer.metadata_df["image_path"].head(12).tolist()
        sample_cols = st.columns(6)
        selected_sample = None
        for idx, sp in enumerate(sample_paths[:6]):
            with sample_cols[idx]:
                st.image(sp, use_container_width=True)
                if st.button(f"Select #{idx+1}", key=f"btn_sample_{idx}"):
                    selected_sample = sp
        
        if selected_sample:
            query_img = Image.open(selected_sample)
            st.success(f"Selected sample image: `{Path(selected_sample).name}`")
        else:
            query_img = Image.open(sample_paths[0])
            st.info(f"Defaulting to sample image: `{Path(sample_paths[0]).name}`")
    else:
        uploaded_file = st.file_uploader("Upload Image File (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            query_img = Image.open(uploaded_file)
            st.image(query_img, caption="Uploaded Query Image", width=260)

    if query_img is not None:
        st.markdown("---")
        if st.button("🚀 Find Nearest Neighbors"):
            results, latency_ms = indexer.search_by_image(query_img, top_k=top_k)
            st.markdown(f"⏱️ **Search Latency:** `{latency_ms:.2f} ms` | Retrieved Top-{len(results)} Nearest Neighbors")
            
            grid_cols = st.columns(4)
            for idx, item in enumerate(results):
                with grid_cols[idx % 4]:
                    score = item["similarity_score"]
                    badge_class = "badge-high" if score >= 0.75 else ("badge-mid" if score >= 0.50 else "badge-low")
                    st.image(item["image_path"], use_container_width=True)
                    st.markdown(f"**Rank #{item['rank']}**")
                    st.markdown(f"<span class='similarity-badge {badge_class}'>Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                    st.caption(f"**Caption:** {item['caption']}")
                    st.caption(f"🏷️ `{item['category']}` | 🎨 `{item['color']}`")

# ---------------------------------------------------------
# TAB 3: HYBRID MULTIMODAL SEARCH
# ---------------------------------------------------------
with tab3:
    st.subheader("Hybrid Multimodal Search (Image + Text Prompt Refinement)")
    st.markdown("Combine visual features from an input image with natural language refinement instructions.")

    col_h1, col_h2 = st.columns([1, 2])
    with col_h1:
        sample_img_path = indexer.metadata_df["image_path"].iloc[0]
        st.image(sample_img_path, caption="Base Reference Image", use_container_width=True)
        base_img = Image.open(sample_img_path)

    with col_h2:
        hybrid_prompt = st.text_input("Refinement Text Prompt:", value="make it yellow supercar")
        alpha = st.slider("Text Weight vs Image Weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        if st.button("⚡ Execute Hybrid Query"):
            results, latency_ms = indexer.search_hybrid(base_img, hybrid_prompt, alpha=alpha, top_k=top_k)
            st.markdown(f"⏱️ **Search Latency:** `{latency_ms:.2f} ms` (Alpha={alpha})")
            
            h_cols = st.columns(4)
            for idx, item in enumerate(results):
                with h_cols[idx % 4]:
                    st.image(item["image_path"], use_container_width=True)
                    st.markdown(f"**Rank #{item['rank']}** (Score: `{item['similarity_score']:.4f}`)")
                    st.caption(f"{item['caption']}")

# ---------------------------------------------------------
# TAB 4: LIVE IMAGE INGESTION ("ADD TO FAISS INDEX")
# ---------------------------------------------------------
with tab4:
    st.subheader("➕ Ingest New Photo into Live FAISS Vector Index")
    st.markdown("Add any new photo into the vector index in real time. It embeds with CLIP and becomes immediately searchable!")

    ingest_source = st.radio("Ingestion Source:", ["Image URL from Web", "Upload Image File"], horizontal=True)
    ingest_img = None
    ingest_url = ""

    col_in1, col_in2 = st.columns([1, 1])
    with col_in1:
        if ingest_source == "Image URL from Web":
            ingest_url = st.text_input("Enter Direct Image URL (JPG/PNG):", placeholder="https://images.unsplash.com/photo-...")
            if ingest_url:
                try:
                    st.image(ingest_url, caption="Image Preview", width=280)
                    ingest_img = ingest_url
                except Exception as e:
                    st.error("Could not load image from this URL. Please check the link.")
        else:
            uploaded_new = st.file_uploader("Choose Image File:", type=["jpg", "jpeg", "png"], key="new_ingest_file")
            if uploaded_new:
                ingest_img = Image.open(uploaded_new)
                st.image(ingest_img, caption="Image Preview", width=280)

    with col_in2:
        new_caption = st.text_input("Photo Caption / Description:", placeholder="e.g. Modern electric motorcycle with neon accents")
        new_cat = st.selectbox("Category:", ["automotive", "bicycles_motorcycles", "footwear", "electronics", "animals", "beverages_and_food", "lifestyle_fashion", "nature_architecture", "custom"])
        new_color = st.text_input("Dominant Color:", value="black")
        new_tags = st.text_input("Keywords / Tags:", placeholder="e.g. motorcycle, electric, futuristic")

        if st.button("🚀 Ingest Photo into Vector Index"):
            if not ingest_img:
                st.error("Please provide an image URL or upload an image file first.")
            elif not new_caption.strip():
                st.error("Please provide a description caption for this image.")
            else:
                with st.spinner("Extracting 512-D CLIP Embedding & Updating FAISS Index..."):
                    try:
                        record = indexer.add_image_to_index(
                            image=ingest_img,
                            caption=new_caption,
                            category=new_cat,
                            color=new_color,
                            tags=new_tags
                        )
                        st.success(f"🎉 Successfully ingested into FAISS Index! New Vector ID: #{record['id']}")
                        st.info("You can now immediately search for this photo in the 'Text → Image' search tab!")
                        st.cache_resource.clear()
                    except Exception as e:
                        st.error(f"Error adding image: {e}")

# ---------------------------------------------------------
# TAB 5: EVALUATION & BENCHMARKS DASHBOARD
# ---------------------------------------------------------
with tab5:
    st.subheader("Quantitative Evaluation & Qualitative Failure Analysis")
    st.markdown("Retrieval benchmarks evaluated over ground-truth annotated test queries.")

    if EVAL_REPORT_PATH.exists():
        with open(EVAL_REPORT_PATH, "r") as f:
            report = json.load(f)
            
        metrics = report["metrics"]
        p_k = metrics["precision_at_k"]
        lat = metrics["latency_stats_ms"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(f"<div class='metric-box'><div class='metric-value'>{p_k['P@1']:.2f}</div><div class='metric-label'>Precision@1</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-box'><div class='metric-value'>{p_k['P@5']:.2f}</div><div class='metric-label'>Precision@5</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-box'><div class='metric-value'>{p_k['P@10']:.2f}</div><div class='metric-label'>Precision@10</div></div>", unsafe_allow_html=True)
        m4.markdown(f"<div class='metric-box'><div class='metric-value'>{metrics['mean_reciprocal_rank_MRR']:.2f}</div><div class='metric-label'>MRR Score</div></div>", unsafe_allow_html=True)
        m5.markdown(f"<div class='metric-box'><div class='metric-value'>{lat['mean_ms']:.1f}ms</div><div class='metric-label'>Mean Latency</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        c_plot1, c_plot2 = st.columns([1, 1])
        with c_plot1:
            if EVAL_PLOT_PATH.exists():
                st.image(str(EVAL_PLOT_PATH), caption="Precision@K and Latency Distributions", use_container_width=True)

        with c_plot2:
            st.markdown("### 🔍 Qualitative Failure Analysis")
            failures = report.get("failure_cases", [])
            for f_idx, fail in enumerate(failures[:3], 1):
                with st.expander(f"Failure Case #{f_idx}: '{fail['query_text']}' (P@5 = {fail['precision_at_5']:.2f})"):
                    st.write(f"**Target Category:** `{fail['target_category']}`")
                    st.write(f"**Failure Explanation:** {fail['failure_explanation']}")
                    st.write("**Top Retrieved Items:**")
                    for tr in fail["top_retrieved"][:3]:
                        st.caption(f"- Rank #{tr['rank']}: ID={tr['id']} | `{tr['caption']}` (Score: {tr['similarity_score']:.4f})")
    else:
        st.warning("Evaluation report not found. Run `python evaluate.py` to generate evaluation metrics.")

# ---------------------------------------------------------
# TAB 6: EMBEDDING SPACE EXPLORER
# ---------------------------------------------------------
with tab6:
    st.subheader("2D Vector Space Projection (PCA / t-SNE)")
    st.markdown("Visualize how CLIP places images and text queries in the shared 512-D embedding space.")

    if indexer.embeddings is not None:
        reduction_method = st.radio("Reduction Method:", ["PCA", "t-SNE"], horizontal=True)
        
        with st.spinner(f"Computing 2D {reduction_method} projection..."):
            coords_2d = project_embeddings_2d(indexer.embeddings, method=reduction_method.lower())
            fig = plot_embedding_space(coords_2d, indexer.metadata_df, title=f"CLIP 512-D Embedding Space ({reduction_method} Projection)")
            st.pyplot(fig)
    else:
        st.info("Numpy embeddings not loaded.")

# ---------------------------------------------------------
# TAB 7: STRUCTURED QUERY AGENT & TRACE INSPECTOR
# ---------------------------------------------------------
with tab7:
    st.subheader("Structured Query Agent")
    st.markdown("Turn a natural-language request into a validated retrieval plan and inspect the tool trace.")

    agent_query = st.text_area(
        "Natural-language request",
        value="Find red sports cars without vintage styling",
        height=90,
        help="Exclusion phrases such as 'without' and 'excluding' activate negative-prompt retrieval.",
    )
    agent_top_k = st.slider("Agent Top-K", min_value=1, max_value=24, value=8, key="agent_top_k")

    if st.button("Run Structured Query", type="primary"):
        try:
            with st.spinner("Planning query, selecting retrieval tool, and recording trace..."):
                agent_result = retrieval_agent.run(agent_query, top_k=agent_top_k)
            st.success(f"Completed request `{agent_result['request_id']}` with `{agent_result['tool']}`.")
            st.json(agent_result["plan"])
            st.code(format_structured_result(agent_result), language="json")
            st.caption(f"Trace written to `{retrieval_agent.trace_logger.trace_path}`. Image bytes are never stored in the trace.")

            for item in agent_result["results"]:
                st.write(
                    f"**#{item['rank']}** {item['caption']} | "
                    f"score `{item['similarity_score']:.4f}` | "
                    f"category `{item['category']}`"
                )
        except ValueError as exc:
            st.error(str(exc))
