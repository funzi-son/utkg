"""
Very simple negative sampling
SNT
"""
import numpy as np
import tensorflow.compat.v1 as tf
FLAGS = tf.flags.FLAGS
# NOTE: negative sampling should be improved
def negative_vals_sampling(all_vals,pos_val,nneg_per_var):
    """
    Sampling values that differet from a positive values
    """
    neg_inds = np.where(all_vals==pos_val)[0] 
    spl_inds = np.random.choice(neg_inds,nneg_per_var)

    return neg_vals[spl_inds]

def negative_sampling_rand(data,nneg_sampl):
    # randomly pick a positive tuple
    sinx = np.random.randint(data.shape[0],size=1)[0]
    # randomly choose a var to change
    vinx = np.random.randint(data.shape[1],size=1)[0]
    # negative sampling
    neg_sampls = negative_tuple_sampling(data,data[sinx,:],vinx,1)
    
    return neg_sampls
    
def negative_tuple_sampling(data,pos_tuple,var_inx,nneg_per_var):
    """
    sample values to make a tuples that aree different from
    the positive tuple
    """

    true_list = None
    for i in range(data.shape[1]):
        if i != var_inx:
            compare = (data[:,i] == pos_tuple[i])
            if true_list is None:
                true_list = compare
            else:
                true_list = np.logical_and(true_list,compare)

    pos_vals = np.unique(data[np.where(true_list)[0],var_inx])
    all_vals = np.unique(data[:,var_inx])
    neg_vals = np.setdiff1d(all_vals,pos_vals)

    nsample = min(nneg_per_var,len(neg_vals))
    
    neg_samples = np.ones((nsample,data.shape[1]))
    #print(neg_samples)
    neg_samples = neg_samples*pos_tuple #need to check
    #print(neg_samples)
    #print(neg_vals)
    select_neg_vals = np.random.choice(neg_vals,nsample)#[:,np.newaxis]
    #print(select_neg_vals)
    neg_samples[:,var_inx] = select_neg_vals 
    #print(neg_samples)
    #input("")
    return neg_samples

"""
def negative_sampling(inds,data,var_inx,nneg_per_sample=1):
    # NO TEST
    all_vals = data[:,var_inx]
    
    neg_samples = np.empty((0,data.shape[1]))
    for i in inds:
        #pos_sampl = data[i,:] # keep dims
        pos_val   = data[i,var_inx]
        neg_vals  = negative_vals_sampling(all_vals,pos_val,nneg_per_sample)
        neg_sampl = copy(data[i,:])#np.zeros((len(pos_vals),data.shape[1]))
        neg_sampl[:,var_inx] = neg_vals

        neg_samples = np.append(neg_samples,neg_sampl,axis=0)
        
    return neg_samples
"""

def negative_sampling(inds,data):
    neg_samples = np.empty((0,data.shape[1]))
    for i in inds:
        for var_inx in range(data.shape[1]):
            pos_tuple   = data[i,:]
            neg_tuples  = negative_tuple_sampling(data,pos_tuple,var_inx,FLAGS.nneg_per_var)
            neg_samples = np.append(neg_tuples,neg_samples,axis=0)

    #print(neg_samples)
    #print(neg_samples.shape)
    #input("")
    return neg_samples
                          
