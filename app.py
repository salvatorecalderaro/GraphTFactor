import streamlit as st
from PIL import Image
from Bio import SeqIO
import io
from utils import identify_device, load_model, load_model_from_file
from prot2graph import create_graph
from GraphTFactor import predict_graph

LOGO_PATH = "logo.jpeg"  

# --- Page Configuration ---
st.set_page_config(page_title="GraphTFactor", page_icon=LOGO_PATH)

sequence = None
sequences = None

device, dev_name = identify_device()

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

input_method = st.radio("Input Method", ["Upload FASTA File", "Paste Sequence"])
sequence = ""

if input_method == "Upload FASTA File":
    fasta_file = st.file_uploader("Upload your protein FASTA file",type=["fasta", "fa", "txt"])

    if fasta_file:
        try:

            fasta_content = io.StringIO(fasta_file.getvalue().decode("utf-8"))

            records = list(SeqIO.parse(fasta_content, "fasta"))

            if len(records) == 0:
                st.error("❌ No sequences found in FASTA file")

            else:
                st.success(f"✅ Loaded {len(records)} protein sequences")

                # salva tutte le sequenze

                sequences = {record.id: str(record.seq) for record in records}

                # visualizzazione

                st.write("### Sequences detected:")

        except Exception as e:

            st.error(f"❌ Error reading FASTA: {e}")

elif input_method == "Paste Sequence":

    pasted_seq = st.text_area(

        "Paste your protein sequence here",

        height=150

    )

    if pasted_seq:

        sequence = (
            pasted_seq
            .replace("\n", "")
            .replace("\r", "")
            .replace(" ", "")
            .upper()
            .strip()
        )

        st.write(f"Sequence length: {len(sequence)} aa")
        
        
if sequence or sequences:
    esm_model = st.selectbox("🧬 Select ESM-2 model", [6,12,30], index=2)
    organism = st.selectbox("🧬 Select Organism", ["All", "Eukaryotic", "Prokaryotic", "Virus"], index=0)
    if st.button("🔍 Predict Transcription Factor Presence"):

        if sequence and esm_model and organism:

            progress_bar = st.progress(0)
            status = st.empty()

            try:
                # Step 1 - Load ESM model
                status.write("🧬 Loading ESM-2 model...")
                progress_bar.progress(10)

                esm_model = int(esm_model)
                model, alphabet = load_model(
                    esm_model,
                    device
                )

                progress_bar.progress(40)


                # Step 2 - Create graph
                status.write("🕸️ Building protein graph...")
                
                graph = create_graph(
                    sequence,
                    model,
                    alphabet,
                    esm_model,
                    device=device,
                    y=0
                )
                
                in_channels=graph.num_node_features

                progress_bar.progress(70)


                # Step 3 - Prediction
                status.write("🤖 Running Graph Neural Network prediction...")
                graphtfmodel = load_model_from_file(
                    organism,
                    esm_model,
                    in_channels,
                    device
                )
                prediction,proba = predict_graph(graphtfmodel, graph, device)

                progress_bar.progress(100)

                status.success("✅ Prediction completed")

                # Result
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.success(f"✅ The protein contains a Transcription Factor with probability {proba:.4f}")
                else:
                    st.error(f"❌ The protein does not contain a Transcription Factor with probability {1-proba:.4f}")


            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")


        else:
            st.warning(
                "⚠️ Please provide a sequence, select ESM model and organism"
            )