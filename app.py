import streamlit as st
from PIL import Image
LOGO_PATH = "logo.jpeg"  

# --- Page Configuration ---
st.set_page_config(page_title="GraphTFactor", page_icon=LOGO_PATH)


# --- Sidebar with Logo and Citation ---
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=200)
    except Exception as e:
        st.warning(f"⚠️ Could not load logo: {e}")

    st.markdown("""
    ---
    ## 📄 Reference

    This app uses a **Graph Neural Network (GNN)** to predict transcription factor presence from protein sequences.

    **Citation:**  
    

    ---
    """)

# --- Title ---
st.title("🧬 GraphTFactor")
st.markdown("**Modeling Protein Sequences as Graphs for Accurate Transcription Factor Prediction**")