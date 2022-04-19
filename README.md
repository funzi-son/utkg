# utkg
Uncertain Temporal Knowledge Graph Reasoning
## Dependencies
- • Python 3.8+
- • Numpy 1.22+
• Tensorflow 1.5+ (tensorflow.compat.v1 in Tensorflow 2.8)
• networkx==2.5.1
• numpy==1.21.1
• pyvis==0.1.9
• streamlit==0.83.0

## Train a model:
run.sh thid_num=<?> phid_num=<?> vhid_num=<?> npred_hid=<?> iter=<?> lr=<?> nneg_per_var=<?> bsize=<?> optimizer=<?>

• thid_num: number of hidden unit in time embedding.
• phid_num: number of hidden unit in predicate embedding.
• vhid_num:number of hidden unit in object embedding.
• npred_hid:number of hidden unit neural predicate.
• iter: max iteration.
• lr:learning rate.
• nneg_per_var:number of negative samples generated from changing values of a variable.
• bsize: mini-batch size.
• optimizer: Optimizer used for training ("adam", "rmsprop").


## Run a demo
demo.sh thid_num=<?> phid_num=<?> vhid_num=<?> npred_hid=<?> iter=<?> lr=<?> nneg_per_var=<?> bsize=<?> opti-
mizer=<?>

where the parameters here should be the same as from the training so that the same model
can be reused for the demonstration
