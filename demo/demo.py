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

from utkg.kbs.kg import GraphKB, GraphDB
from utkg.models.utkg import UTKG

import tensorflow.compat.v1 as tf

from pathlib import Path

home = str(Path.home())
flags = tf.app.flags
FLAGS = flags.FLAGS

# Modelling params                                                                                                             
flags.DEFINE_integer("thid_num",10,"number of hidden unit in time embedding")
flags.DEFINE_integer("phid_num",10,"number of hidden unit in predicate embedding")
flags.DEFINE_integer("vhid_num",10,"number of hidden unit in predicate embedding")



# Learning params                                                                                                              
flags.DEFINE_integer("iter",100,"max iteration")
flags.DEFINE_float("lr",0.001,"learning rate")
flags.DEFINE_integer("nneg_per_var",2,"number of negative samples generated from changing values of a variable")
flags.DEFINE_integer("bsize",32,"mini-batch size")
flags.DEFINE_string("optimizer","adam","training alg")

with  tf.Graph().as_default():
    kb = GraphKB("../data/footballdb/")
    dataset = GraphDB(kb)
    model = UTKG(kb)

    print("[UTKG] Complete constructing models. Start loading ..")

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    # Run the model                                                                                                        
    model.load(session,"../examples/footballdb/checkpoint/utkg.ckpt")
    

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

#for var_name in self.kb.classes:
#    print

sj_list = [model.kb.objects[i] for i in model.kb.classes["uhuman"]]
ob_list = [model.kb.objects[i] for i in model.kb.classes["uteam"]]
rl_list = ["playsFor"]
yr_list = [str(i) for i in range(1950,2017)]

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

st.write('IF wasBornOn(X,T1) AND before(T2,T1) THEN NOT playsFor(X,Y,T2)')

click = st.button("Verify", key=None, help=None, on_click=None, args=None, kwargs=None)


if click:
    st.write("Certainty of query")

    
    vid = list(model.kb.objects.values()).index(selected_subject)
    sid = list(model.kb.objects.keys())[vid]

    vid = list(model.kb.objects.values()).index(selected_object)
    oid = list(model.kb.objects.keys())[vid]
   
    query = {"name":"playFor",
             "uhuman":sid,
             "uteam":oid,
             "tY":int(selected_year)}
    
    confidence = model.infer(session,query)
    
    st.write(str(confidence) + ": NOT playsFor("+selected_subject+","+selected_object+","+selected_year+")")

    #### Explainability
    wbo    = model.npreds["wasBornOn"]
    n = wbo.get_truth(session,np.array([[sid,int(i)] for i in yr_list])) 
    x = [int(i) for i in yr_list]
    
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(x, n,label="wasBornOn("+selected_object+",T1)",color="r")
    #ax.plot([1993,1993],[0,max(n)],"--")
    #ax.text(0.65, 0.01, '1993-01-08', horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel("Year")
    ax.set_ylabel("wasBornOn")
    #ax.legend()


    before = model.npreds["before"]
    y = before.truth_range(int(selected_year),[int(i) for i in yr_list])
    axis_2 = ax.twinx()
    axis_2.plot(x,y,label="before("+selected_year+",T1)")
    axis_2.set_ylabel("before")
    #axis_2.legend()

    #fig.legend(loc=2)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    

# Footer
st.markdown(
    """
    <br>
    <h6>The project is funded by Defense Science Institute.</h6>
    """, unsafe_allow_html=True
    )
