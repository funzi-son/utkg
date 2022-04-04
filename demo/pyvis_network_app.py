import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from io import BytesIO
#from utkg.kbs.kg import GraphKB
#from utkg.data.

# Read dataset (CSV)
df_interact = pd.read_csv('data/processed_drug_interactions.csv')

# Set header title
st.title('Human-AI Knowledge Interface')

st.markdown(
    """
    <br>
    <h6>Son N. Tran (<a href="Axirv" target="_blank">paper</a>, <a href="github" target="_blank">code</a>)</h6>
    """, unsafe_allow_html=True
    )


# Define list of selection options and sort alphabetically
#kb = GraphKB("")

sj_list = ["Isaiah Crowell"]
ob_list = ["Cleveland Browns"]
rl_list = ["playsFor","wasBornOn"]
yr_list = ["1990","2005","2006","2016"]

sj_list.sort()
ob_list.sort()
yr_list.sort()
st.write("Query: subject relation object time)")
name_cols = st.columns(4)

selected_subject = name_cols[0].selectbox("subject",sj_list)

# Implement multiselect dropdown menu for option selection (returns a list)
#selected_subject = st.multiselect('Subject', sj_list)
#selected_subject = st.selectbox('subject', sj_list)
# Implement multiselect dropdown menu for option selection (returns a list)
#selected_relationss = st.multiselect('Relation', rl_list)
selected_relationss = name_cols[1].selectbox('relation', rl_list)
# Implement multiselect dropdown menu for object
#selected_object = st.multiselect('Object', ob_list)
selected_object = name_cols[2].selectbox('object', ob_list)

#
#selected_year = st.multiselect('Object', yr_list)
selected_year = name_cols[3].selectbox('time', yr_list)

sentence = st.text_input('Input knowledge')

click = st.button("Verify", key=None, help=None, on_click=None, args=None, kwargs=None)


if click:
    st.write("Confidence of knowledge")
    st.write("0.98: IF wasBornOn(X,T1) AND after(T1,T2) THEN NOT playsFor(X,Y,T2)")
    st.write("Certainty of query")
    st.write("0.97: NOT playsFor(Isaiah Crowell,Cleveland Browns,1990)")
    mu = 1993
    variance = 0.1
    sigma = math.sqrt(variance)
    x = [i for i in range(1950,2020)]
    fig, ax = plt.subplots(figsize=(5,4))
    n = stats.norm.pdf(x, mu, sigma)
    n = (n-min(n))/(max(n)-min(n))
    y = [1/(1+np.exp(-(i-1990))) for i in x]
    ax.plot(x, n,label="wasBornOn(Isaiah Crowell,T1)",color="r")
    ax.plot([1993,1993],[0,max(n)],"--")
    ax.text(0.65, 0.01, '1993-01-08', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel("Year")
    ax.set_ylabel("wasBornOn")
    #ax.legend()
    axis_2 = ax.twinx()
    axis_2.plot(x,y,label="after(1993-01-08,T2)")
    axis_2.set_ylabel("after")
    #axis_2.legend()

    #fig.legend(loc=2)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    #df_sselect = df_interact.loc[df_interact['subject'].isin(selected_subject) | \
    #                            df_interact['object'].isin(selected_object)]
    #df_select = df_select.reset_index(drop=True)

    # Create networkx graph object from pandas dataframe
    #G = nx.from_pandas_edgelist(df_select, 'subject', 'object', 'weight')

    # Initiate PyVis network object
    #mynet = Network(
    #                   height='400px',
    #                   width='100%',
    #                   bgcolor='#222222',
    #                   font_color='white'
    #                  )

    # Take Networkx graph and translate it to a PyVis graph format
    #drug_net.from_nx(G)

    # Generate network with specific layout settings
    #drug_net.repulsion(
    #                    node_distance=420,
    #                    central_gravity=0.33,
    #                    spring_length=110,
    #                    spring_strength=0.10,
    #                    damping=0.95
    #                   )

    # Save and read graph as HTML file (on Streamlit Sharing)
    #try:
    #    path = '/tmp'
    #    drug_net.save_graph(f'{path}/pyvis_graph.html')
    #    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    #except:
    #    path = '/html_files'
    #    drug_net.save_graph(f'{path}/pyvis_graph.html')
    #    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    #components.html(HtmlFile.read(), height=435)

# Footer
st.markdown(
    """
    <br>
    <h6>The project is funded by Defense Science Institute.</h6>
    """, unsafe_allow_html=True
    )
