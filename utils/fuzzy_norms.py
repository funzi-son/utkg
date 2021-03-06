"""
t-norm computation
#Son N. Tran
"""
import numpy as np

def literal(sign,value):
    """
    1-x: NOT x
    x  : x
    """
    return (1-sign) + (2*sign-1)*value
    
def prod_norm_conj(clause):
    """
    x*y
    product norm for conjunctive clause
    """
    #print(clause)
    truth = 1
    for i in range(len(clause)):
        print((clause[i][0],clause[i][1]))
        truth *= literal(clause[i][0],clause[i][1])

    #print(truth)
    return truth

def prod_norm_test():
    """
    a AND b AND c
    """
    data = None
    # Truth values
    t = []
    # tnorm
    cjclause = [[1,-1],[1,-1],[1,-1]]
    k =  []
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                assignment = np.array([[a,b,c]])
                if data is None:
                    data = assignment
                else:
                    data = np.append(data,assignment,axis=0)

    
                # compute truth
                if a==1 and b==1 and c==1:          
                    t.append(1)
                else:
                    t.append(0)

                # prod
                cjclause[0][1] = a
                cjclause[1][1] = b
                cjclause[2][1] = c

                k.append(prod_norm(cjclause))

    print(t)
    print(k)

def lukasiewicz_min(clause):
    """
    lukasiewicz_min = min(x+y,1)

    clause is a 2-dimensional array where clause[i][0] is the
    sign of literal i (0 negative,1 positive) and the clause[i][1] 
    is the literal's value 
    """
    #print(clause)
    lmin = 0
    for i in range(len(clause)):
        print((clause[i][0],clause[i][1]))
        lmin += literal(cjclause[i][0],cjclause[i][1])

    lmin  =  min(lmin,1)
    #print(lmin)
    #input("")
    return lmin

def lukasiewicz_min_test():
    """
    a AND b THEN c
    """
    data = None
    # Truth values
    t = []
    # tnorm
    clause = [[0,-1],[0,-1],[1,-1]]
    k =  []
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                assignment = np.array([[a,b,c]])
                if data is None:
                    data = assignment
                else:
                    data = np.append(data,assignment,axis=0)

    
                # compute truth
                if a==1 and b==1 and c==0:          
                    t.append(0)
                else:
                    t.append(1)

                # t-norm
                clause[0][1] = a
                clause[1][1] = b
                clause[2][1] = c

                k.append(lukasiewicz_min(clause))

    print(t)
    print(k)


if __name__=="__main__":
    prod_norm_test()
