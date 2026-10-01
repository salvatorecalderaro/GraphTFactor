import streamlit as st
from PIL import Image
from collections import Counter
import json
import subprocess
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
st.set_page_config(page_title="GraphTFactor",page_icon=LOGO_PATH,layout="wide")


st.markdown(
"""
<style>


.main{
    background-color:#fafafa;
}


h1{
    color:#163a5f;
}


.card{

background:white;
padding:25px;
border-radius:15px;
box-shadow:
0px 3px 12px rgba(0,0,0,0.12);

margin-top:15px;

}



[data-testid="metric-container"]{

background:white;
padding:15px;
border-radius:12px;

box-shadow:
0px 2px 8px rgba(0,0,0,0.10);

}



.tf{

color:#008000;
font-size:28px;
font-weight:700;

}



.nontf{

color:#cc0000;
font-size:28px;
font-weight:700;

}


.small-title{

font-size:18px;
font-weight:600;
color:#444;

}


</style>
""",
unsafe_allow_html=True
)


with st.sidebar:
    try:
        logo=Image.open(LOGO_PATH)
        st.image(logo,width=220)

    except:
        st.warning("Logo not found")



    st.markdown(
    """

    ---
    
    ## 🧬 GraphTFactor

    Graph Neural Network framework
    for transcription factor prediction
    from protein sequences.


    ### Pipeline

    Protein sequence

    ↓

    ESM-2 embedding

    ↓

    Protein graph

    ↓

    GNN classifier


    ---

    """

    )

st.title("🧬 GraphTFactor")


st.markdown(
"""
### Protein graph learning for transcription factor prediction
"""
)



sequence = st.text_area( "Paste protein sequence", height=180)


sequence = (
    sequence
    .replace("\n","")
    .replace("\r","")
    .replace(" ","")
    .upper()
    .strip()
)


if sequence:
    VALID_AA=set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    invalid=set(sequence)-VALID_AA
    if invalid:
        st.error(f"Invalid residues: {invalid}")
        st.stop()


    st.subheader("🧬 Protein information")
    aa=Counter(sequence)
    c1,c2,c3=st.columns(3)


    c1.metric("Length",f"{len(sequence)} aa")
    c2.metric("Unknown residues", aa.get("X",0))
    c3.metric("Unique residues",len(aa))


    st.markdown("### Amino acid composition")


    aa_df=pd.DataFrame(
        aa.items(),
        columns=[
            "Residue",
            "Count"
        ]
    )


    fig=px.bar(aa_df,x="Residue",y="Count",template="plotly_white")
    st.plotly_chart(fig,width="stretch")


    st.subheader("⚙️ Prediction settings")


    col1,col2=st.columns(2)


    with col1:
        esm_model=st.selectbox("ESM-2 model",[6,12,30])


    with col2:
        organism=st.selectbox("Organism",["All","Virus","Eukaryotic","Prokaryotic"])

    if st.button("🔍 Predict Transcription Factor"):
        progress = st.progress(5)
        status = st.empty()

        try:
            status.info("🧬 Loading ESM-2 and predicting on MPS...")
            worker = Path(__file__).parent / "predict_worker.py"
            completed = subprocess.run(
                [sys.executable, str(worker), sequence, str(esm_model), organism],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
            )

            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(
                    f"Prediction worker exited with code {completed.returncode}. "
                    f"{details[-2500:]}"
                )

            result_line = next(
                (line for line in reversed(completed.stdout.splitlines())
                 if line.startswith("RESULT_JSON:")),
                None,
            )
            if result_line is None:
                raise RuntimeError("Prediction worker returned no result.")

            result = json.loads(result_line.removeprefix("RESULT_JSON:"))
            progress.progress(65)
            status.info("🕸️ Preparing graph visualization...")

            graph_data = result["graph"]
            G = nx.Graph()
            G.add_nodes_from(range(graph_data["nodes"]))
            G.add_edges_from(map(tuple, graph_data["edges"]))
            pos = nx.spring_layout(G, seed=42)

            edge_x, edge_y = [], []
            for source, target in G.edges():
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]

            g1, g2, g3 = st.columns(3)
            g1.metric("Nodes", graph_data["nodes"])
            g2.metric("Edges", graph_data["edge_count"])
            g3.metric("Features", graph_data["features"])

            fig_graph = go.Figure()
            fig_graph.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines"))
            fig_graph.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers", marker=dict(size=8)
            ))
            fig_graph.update_layout(
                title="Protein graph", template="plotly_white", height=500,
                showlegend=False,
            )
            st.plotly_chart(fig_graph)

            progress.progress(85)
            prediction = result["prediction"]
            proba = result["probability"]
            confidence = proba if prediction == 1 else 1 - proba
            status.success("Prediction completed")

            if prediction == 1:
                label = "✅ Transcription Factor"
                css_class = "tf"
            else:
                label = "🚩 Non Transcription Factor"
                css_class = "nontf"

            st.markdown(
                f"""
                <div class="card">
                <div class="{css_class}">{label}</div>
                Confidence: {confidence * 100:.2f} %
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": "Prediction confidence (%)"},
                gauge={"axis": {"range": [0, 100]}},
            ))
            fig_conf.update_layout(height=300)
            st.plotly_chart(fig_conf)
            progress.progress(100)

        except Exception as e:
            status.error(f"Prediction error: {e}")
