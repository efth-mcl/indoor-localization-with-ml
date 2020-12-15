from itertools import product
from spektral.utils import localpooling_filter
import tensorflow as tf
import numpy as np

def MyLer(a,Cj,n):
    C = []
    for j in range(n.shape[1]):
        nn = tf.eye(n.shape[0],n.shape[0])*n[:,j]

        c = tf.matmul(tf.matmul(a,Cj[:,:,j]),nn)
        
        C.append([c])
    C = tf.concat(C,0)
    return tf.transpose(C), tf.transpose(tf.reduce_sum(C,axis=2)), tf.reduce_sum(C,axis=0)



def Ci(A, Cold):
    Cnew = np.zeros(Cold.shape)
    freevals = list(product(
        list(range(Cold.shape[0])),
        list(range(Cold.shape[1])),
        list(range(Cold.shape[2])),
        list(range(Cold.shape[3]))
    ))
    dummyvals = list(product(
        # bd
        list(range(A.shape[0])),
        list(range(Cold.shape[0])),

        # kd
        list(range(A.shape[2]))
    ))
    for b, i0, j0, f in freevals:
        s = 0
        for bda, bdc, kd  in dummyvals:
            if b == bda and b == bdc:
                s += A[bda,i0,kd]*Cold[bdc,kd,j0,f]
        Cnew[b, i0, j0, f] = s

    Cnew = tf.cast(Cnew,tf.float32)

    return Cnew


def K(A, X):
    k = np.zeros((A.shape+[X.shape[2]]))
    freevals = list(product(
                    list(range(A.shape[0])),
                    list(range(A.shape[1])),
                    list(range(A.shape[2])),
                    list(range(X.shape[2])),
                ))
    dummyvals = list(product(
                    list(range(A.shape[0])),
                    list(range(X.shape[0])),
                    list(range(A.shape[2])),
                    list(range(X.shape[1])),
                ))

    for b, i0,j0,f in freevals:
        s = 0
        
        
        for bda, bdx, jda,jdx in dummyvals:
            if b == bda and b == bdx and j0 == jda and j0 == jdx:
                s+= A[bda,i0,jda]*X[bdx,jdx,f]
                

        k[b, i0,j0,f] = s
    
    k = tf.cast(k,tf.float32)
    return k

def phi(h):
    return tf.nn.relu(h)




def myphi0(C):
    Cnew = np.zeros(C.shape)
    freevlas = list(product(
                    list(range(C.shape[0])),
                    list(range(C.shape[1])),
                    list(range(C.shape[2])),
                    list(range(C.shape[3]))
                ))
    dummyvals = list(product(
                    list(range(C.shape[1])),
                    list(range(C.shape[2])),
                ))
    dummyvals = list(range(C.shape[2]))
    for b, i, j, f in freevlas:
        s = tf.reduce_sum([C[b,i,jd,f] for jd in dummyvals])
        sexp = tf.reduce_sum([tf.nn.tanh(C[b,i,jd,f]) for jd in dummyvals])

        if tf.abs(s) > 1e-5:
            Cnew[b,i,j,f] = phi(s)*C[b,i,j,f]/(s)
        else:
            sexp = tf.reduce_sum([tf.exp(C[b,i,jd,f]) for jd in dummyvals])
            Cnew[b,i,j,f] = phi(s)*tf.exp(C[b,i,j,f])/(sexp)
    
    Cnew = tf.cast(Cnew,tf.float32)
    return Cnew

def myphi1(C):
    return myphi0(C)
    Cnew = np.zeros(C.shape)
    freevlas = list(product(
                    list(range(C.shape[0])),
                    list(range(C.shape[1])),
                    list(range(C.shape[2])),
                    list(range(C.shape[3]))
                ))
    dummyvals = list(product(
                    list(range(C.shape[1])),
                    list(range(C.shape[2])),
                ))
    dummyvals = list(range(C.shape[1]))
    for b, i, j, f in freevlas:
        s = 0
        for i0d in dummyvals:
            s += C[b,i0d,j,f]

        Cnew[b,i,j,f] = phi(s)/C.shape[1]
    
    Cnew = tf.cast(Cnew,tf.float32)
    return Cnew

a0 = tf.cast([
    [1,1,0,0,0,1],
    [0,1,1,0,0,0],
    [0,0,1,1,0,0],
    [0,0,0,1,1,0],
    [0,0,0,0,1,1],
    [0,0,0,0,0,1]
],tf.float32)
a0 = tf.reshape(a0,[1,6,6])
A = tf.concat([a0,a0],axis=0)
A = localpooling_filter(A.numpy())
A = tf.cast(A, tf.float32)

w0 = 200*tf.keras.initializers.GlorotUniform()(shape=[4, 5])


X0 = tf.keras.initializers.GlorotUniform()(shape=[2,6, 4])

def myeye(N):
    return tf.cast([[[1 if i==j and j==w  else 0 for i in range(N)] for j in range(N)] for w in range(N)], tf.float32)

def NewK(A, X):
    D = myeye(A.shape[0])
    U = myeye(A.shape[1])
    C = tf.tensordot(D,A,[[1],[0]])
    C = tf.tensordot(C,U,[[3],[1]])
    C = tf.tensordot(C,X,[[1,4],[0,1]])
    return C


def NewCi(A, C):
    U = myeye(A.shape[0])
    Cnew = tf.tensordot(U,A,[[1],[0]])
    Cnew = tf.tensordot(Cnew,C,[[1, 3],[0,1]])
    return Cnew

def to20RNNphi(C):
    I = tf.reshape(tf.eye(5,5),(1,5,5))
    I = tf.concat([I, tf.zeros((5,5,5))],axis=0)
    I = tf.concat([tf.zeros((6,1,5)), I],axis=1)
    Cto2ORNN = tf.reduce_sum(C,axis=3)
    Cto2ORNN = tf.tensordot(Cto2ORNN,I,[[1,2],[0,1]])
    return tf.nn.relu(1-tf.pow(10,-Cto2ORNN))

C0 = NewK(A,X0)
CC0 = NewK(A,X0)

h0 = tf.matmul(A,X0)

h0 = tf.matmul(h0,w0)
C0 = tf.matmul(C0,w0)
CC0 = tf.matmul(CC0,w0)


assert np.testing.assert_allclose(h0.numpy(), tf.reduce_sum(C0,axis=2).numpy(), rtol=1e-5, atol=0)



h0 = phi(h0)
C0 = myphi0(C0)
CC0 = myphi1(CC0)
print(h0)
print(tf.reduce_sum(C0,axis=2))

CC0s = tf.reduce_sum(CC0,axis=3)
print(tf.nn.relu(1-tf.pow(10,-CC0s[:,0,1:6])))
print(to20RNNphi(CC0))


w1 = tf.keras.initializers.GlorotUniform()(shape=[5, 5])
h1 = tf.matmul(A,h0)
C1 = NewCi(A, C0)
CC1 = NewCi(A, CC0)

h1 = tf.matmul(h1,w1)
C1 = tf.matmul(C1,w1)
CC1 = tf.matmul(CC1,w1)



h1 = phi(h1)
C1 = myphi0(C1)
CC1 = myphi1(CC1)
print(h1)
print(tf.reduce_sum(C1,axis=2))

CC1s = tf.reduce_sum(CC1,axis=3)
print(tf.nn.relu(1-tf.pow(10,-CC1s[:,0,1:6])))
print(to20RNNphi(CC1))

w2 = tf.keras.initializers.GlorotUniform()(shape=[5, 5])
C2 = NewCi(A,C1)
CC2 = NewCi(A,CC1)
h2 = tf.matmul(A,h1)


h2 = tf.matmul(h2, w2)
C2 = tf.matmul(C2,w2)
CC2 = tf.matmul(CC2,w2)


h2 = phi(h2)
C2 = myphi0(C2)
CC2 = myphi1(CC2)
print(h2)
print(tf.reduce_sum(C2,axis=2))

CC2s = tf.reduce_sum(CC2,axis=3)
print(tf.nn.relu(1-tf.pow(10,-CC2s[:,0,1:6])))
print(to20RNNphi(CC2))
print(CC2s[:,0,1:6])
