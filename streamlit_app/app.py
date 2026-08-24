import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from graphtfactor.utils import identify_device, load_model, load_model_from_file,create_nx_graph
from graphtfactor.graph import create_graph
from graphtfactor.model import predict_graph
from pathlib import Path

LOGO_PATH = Path(__file__).parent / "assets" / "logo.jpeg"
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


device,_ = identify_device()



@st.cache_resource
def cached_load_esm(esm_model,device):
    model,alphabet = load_model(esm_model,device)
    return model,alphabet



@st.cache_resource
def cached_load_gnn(organism,esm_model,in_channels,device):
    model = load_model_from_file(organism,esm_model,in_channels,device)
    return model

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
    st.plotly_chart(fig,use_container_width=True)


    st.subheader("⚙️ Prediction settings")


    col1,col2=st.columns(2)


    with col1:
        esm_model=st.selectbox("ESM-2 model",[6,12,30])


    with col2:
        organism=st.selectbox("Organism",["All","Virus","Eukaryotic","Prokaryotic"])

    if st.button("🔍 Predict Transcription Factor"):
        progress=st.progress(0)
        status=st.empty()
        
        try:
            status.info("🧬 Loading ESM-2...")

            esm,alphabet = cached_load_esm(esm_model, device)
            
            progress.progress(30)

            status.info("🕸️ Building protein graph...")


            graph=create_graph(
                sequence,
                esm,
                alphabet,
                esm_model,
                device=device,
                y=0
            )


            progress.progress(55)

            st.subheader("🕸️ Graph information")

            g1,g2,g3=st.columns(3)

            g1.metric("Nodes",graph.num_nodes)
            g2.metric("Edges",graph.num_edges)
            g3.metric("Features",graph.num_node_features)


            
            G = create_nx_graph(graph)
            pos=nx.spring_layout(G,seed=42)


            edge_x=[]
            edge_y=[]


            for e in G.edges():

                x0,y0=pos[e[0]]
                x1,y1=pos[e[1]]


                edge_x += [
                    x0,
                    x1,
                    None
                ]

                edge_y += [
                    y0,
                    y1,
                    None
                ]



            node_x=[]
            node_y=[]


            for n in G.nodes():

                x,y=pos[n]

                node_x.append(x)
                node_y.append(y)



            fig_graph=go.Figure()


            fig_graph.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines"
                )
            )


            fig_graph.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker=dict(
                        size=8
                    )
                )
            )


            fig_graph.update_layout(
                title="Protein graph",
                template="plotly_white",
                height=500,
                showlegend=False
            )


            st.plotly_chart(fig_graph,)



            progress.progress(75)


            status.info("🤖 Running GNN..."
            )


            gnn=cached_load_gnn(
                organism,
                esm_model,
                graph.num_node_features,
                device
            )


            prediction,proba=predict_graph(
                gnn,
                graph,
                device
            )


            progress.progress(100)


            status.success(
                "Prediction completed"
            )



            # ==========================
            # Result
            # ==========================


            if prediction==1:

                confidence=proba


                st.markdown(
                f"""
                <div class="card">

                <div class="tf">
                ✅ Transcription Factor
                </div>

                Confidence:
                {confidence*100:.2f} %

                </div>
                """,
                unsafe_allow_html=True
                )


            else:


                confidence=1-proba


                st.markdown(
                f"""
                <div class="card">

                <div class="nontf">
                🚩 Non Transcription Factor
                </div>

                Confidence:
                {confidence*100:.2f} %

                </div>
                """,
                unsafe_allow_html=True
                )



            # Gauge


            fig_conf=go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence*100,
                    title={
                        "text":
                        "Prediction confidence (%)"
                    },
                    gauge={
                        "axis":{
                            "range":[0,100]
                        }
                    }
                )
            )


            fig_conf.update_layout(
                height=300
            )


            st.plotly_chart(
                fig_conf,
            )



        except Exception as e:


            st.error(
                f"Prediction error: {e}"
            )