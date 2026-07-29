import streamlit as st
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt


st.title("🧬 GraphTFactor - Important Motifs")


seq = st.text_area(
    "Protein sequence",
    "MELIRIAMKKDLENDNSLMNKWATVAGLKNPNPLY"
)

seq = seq.replace("\n","").upper()


# parameters

motif_length = st.slider(
    "Motif length",
    5,
    30,
    10
)

top_k = st.slider(
    "Number of motifs",
    1,
    10,
    5
)


if st.button("Extract motifs"):


    # Dummy Integrated Gradients
    np.random.seed(42)

    scores = np.abs(
        np.random.randn(len(seq))
    )


    # sliding window score

    windows = []

    for i in range(
        len(seq)-motif_length+1
    ):

        motif_score = np.mean(
            scores[i:i+motif_length]
        )

        windows.append(
            (
                i,
                motif_score
            )
        )


    # rank motifs

    windows = sorted(
        windows,
        key=lambda x:x[1],
        reverse=True
    )


    selected=[]


    for pos,score in windows:

        motif = seq[
            pos:pos+motif_length
        ]

        # avoid overlapping motifs

        overlap=False

        for old_pos,_ in selected:

            if abs(pos-old_pos)<motif_length:
                overlap=True


        if not overlap:

            selected.append(
                (pos,score)
            )


        if len(selected)==top_k:
            break



    st.subheader("Important motifs")


    for pos,score in selected:


        motif = seq[
            pos:pos+motif_length
        ]


        st.write(
            f"""
            Position: {pos+1}
            
            Motif: **{motif}**
            
            Importance: {score:.3f}
            """
        )


        # sequence logo for single motif

        data=[]

        alphabet="ACDEFGHIKLMNPQRSTVWY"


        for aa in motif:

            row={
                x:0
                for x in alphabet
            }

            row[aa]=score

            data.append(row)


        df=pd.DataFrame(data)


        fig,ax=plt.subplots(
            figsize=(8,2)
        )


        logomaker.Logo(
            df,
            ax=ax,
            color_scheme="NajafabadiEtAl2017"
        )


        ax.axis("off")


        st.pyplot(fig)