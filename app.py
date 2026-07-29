import streamlit as st
from PIL import Image
from collections import Counter
from utils import identify_device, load_model, load_model_from_file
from prot2graph import create_graph
from GraphTFactor import predict_graph


# ==========================
# Configuration
# ==========================

LOGO_PATH = "logo.jpeg"

st.set_page_config(
    page_title="GraphTFactor",
    page_icon=LOGO_PATH,
    layout="centered"
)


# ==========================
# Device (hidden)
# ==========================

device, _ = identify_device()


# ==========================
# Cache models
# ==========================

@st.cache_resource
def cached_load_esm(esm_model, device):
    model, alphabet = load_model(esm_model, device)
    return model, alphabet



@st.cache_resource
def cached_load_gnn(organism,esm_model,in_channels,device):
    model = load_model_from_file(organism,esm_model,in_channels,device)
    return model



# ==========================
# Sidebar
# ==========================

with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo,width=200)

    except Exception:
        st.error(f"❌ Logo not found: {LOGO_PATH}")


    st.markdown(
        """
        ---
        ## 📄 Reference

        **GraphTFactor**

        Graph Neural Network framework for
        transcription factor prediction from
        protein sequences.

        ---
        """
    )



# ==========================
# Title
# ==========================

st.title("🧬 GraphTFactor")

st.markdown(
    """
    **Modeling protein sequences as graphs for
    transcription factor prediction**
    """
)



# ==========================
# Sequence Input
# ==========================

sequence = st.text_area("Paste protein sequence",height=180)


sequence = (
    sequence
    .replace("\n", "")
    .replace("\r", "")
    .replace(" ", "")
    .upper()
    .strip()
)



# ==========================
# Validation
# ==========================

if sequence:
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    invalid = set(sequence) - VALID_AA

    if invalid:

        st.error(f"Invalid residues detected: {invalid}")
        st.stop()



    # ==========================
    # Protein information
    # ==========================

    st.subheader("🧬 Protein information")


    aa = Counter(sequence)
    c1, c2 = st.columns(2)

    c1.metric("Length",f"{len(sequence)} aa")
    c2.metric("Unknown",aa.get("X", 0))



    # ==========================
    # Settings
    # ==========================

    st.subheader("⚙️ Prediction settings")


    esm_model = st.selectbox("ESM-2 model",[6,12,30],index=0)

    organism = st.selectbox(
        "Organism",
        [
            "All",
            "Virus",
            "Eukaryotic",
            "Prokaryotic"
        ],
        index=0
    )



    # ==========================
    # Prediction
    # ==========================

    if st.button("🔍 Predict Transcription Factor"):
        progress = st.progress(0)
        status = st.empty()


        try:
            status.info("🧬 Loading ESM-2 model...")
            progress.progress(20)

            esm_model = int(esm_model)
            esm, alphabet = cached_load_esm(esm_model,device)
            
            # Graph
            status.info( "🕸️ Building protein graph...")

            progress.progress(50)
            graph = create_graph(
                sequence,
                esm,
                alphabet,
                esm_model,
                device=device,
                y=0
            )



            # Graph information

            st.subheader( "🕸️ Graph information")
            g1, g2, g3 = st.columns(3)


            g1.metric("Nodes",graph.num_nodes)
            g2.metric("Edges",graph.num_edges)
            g3.metric("Features", graph.num_node_features)

            progress.progress(70)

            # GNN prediction
            status.info("🤖 Running Graph Neural Network...")


            gnn = cached_load_gnn(organism,esm_model,graph.num_node_features,device)

            prediction, proba = predict_graph(gnn,graph,device)
            progress.progress(100)

            status.success("✅ Prediction completed")
            # ==========================
            # Result
            # ==========================

            st.subheader("Prediction result")

            if prediction == 1:
                result_label = ("✅ Transcription Factor")
                confidence = proba
            else:
                result_label = ("🚩 Non-Transcription Factor")
                confidence = 1 - proba



            r1, r2 = st.columns(2)


            with r1:
                st.markdown(
                    f"""
                    <div style="
                    font-size:18px;
                    font-weight:600;
                    ">
                    Class<br>
                    {result_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r2:
                st.metric("Confidence",f"{confidence*100:.2f}%")

            st.progress(float(confidence))



            if prediction == 1:
                st.success("✅ The protein contains a Transcription Factor")

            else:
                st.error("🚩 The protein does NOT contain a Transcription Factor")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")