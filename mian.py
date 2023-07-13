import numpy as np
#initialization probability

def pfunction(received, sigma=1):
    return 1/(1 + np.exp(2*received/sigma**2))

# check node update mci→vj=2×atanh(∏vb∈N(ci)\vjtanh(2Lvb→ci))
def check2bit(qNeg, H_mask):
    shape = qNeg.shape# 3,7
    rPos,rNeg = np.zeros(shape),np.zeros(shape)
    qTmp = qNeg[H_mask].reshape(shape[0],4) #4 is the column weight
    qTmp = 1-2*qTmp
    colProduct = np.prod(qTmp, axis=1)
    rTmpPos = 0.5 + 0.5*np.divide(np.full(qTmp.shape, colProduct.reshape(shape[0],1)),qTmp)#
    rTmpNeg = 1-rTmpPos
    rPos[H_mask] = rTmpPos. flatten()
    rNeg[H_mask] = rTmpNeg.flatten()
    return rPos, rNeg

#bit node update Lvj→ci=∑ca∈N(vj)\cimca→vj + Cvj
def bit2check(rPos, receivedPzero, neg=False): 
    rPosMask = rPos == 0
    rPos[rPosMask] = 1
    rowProductPos = np.prod(rPos,axis=0)
    rPostmp = np. divide(np. full(rPos. shape, rowProductPos), rPos)
    rPos[rPosMask] = 0
    rPostmp[rPosMask] = 0
    if not neg:
        qPostmp2 = np.multiply(1-receivedPzero, rPostmp)
    else:
        qPostmp2 = np.multiply(receivedPzero, rPostmp)
    return rowProductPos, qPostmp2

#Sum product decoding algorithm
def sumProduct(received, H, sigma, maxiter=10):
    #initial
    decoded = np. zeros(received. shape[0])
    receivedP = pfunction(received,sigma)
    receivedP = receivedP. reshape(-1, H. shape[1])
    #loop words
    for j in range(len(receivedP)):
        receivedPzero = receivedP[j]
        qNeg = np.multiply(receivedPzero,H) #3,7
        #Pass information from check nodes to bit nodes
        H_mask = H != 0
        k = np.zeros(qNeg.shape)
        for _ in range(maxiter):
            rPos, rNeg = check2bit(qNeg,H_mask)
            # bit nodes to check nodes
            rowProductPos, qPostmp2 = bit2check(rPos, receivedPzero)
            rowProductNeg, qNegtmp2 = bit2check(rNeg, receivedPzero, neg=True)
            k[H_mask] = np.divide(1,qPostmp2[H_mask] + qNegtmp2[H_mask])
            qNeg = np.multiply(k,qNegtmp2)
        QtmpPos = np.multiply(1-receivedPzero, rowProductPos)
        QtmpNeg = np.multiply(receivedPzero, rowProductNeg)
        K = np.divide(1,QtmpPos + QtmpNeg)
        QPos = np.multiply(K,QtmpPos)
        decoded[j*H.shape[1]:j*H.shape[1] + H.shape[1]] = QPos>0.5
    return decoded





def vec_dec2bin(n):
    return [str(i%2) for i in n] 



A = np.array([[1,1,1,0],[0,1,1,1],[1,0,1,1]])
H = np.concatenate((A,np.eye(3)),axis=1)# check matrix
G = np.concatenate((np.eye(4),A.T),axis=1)#generate matrix
Kdec = np.random.randint(0, 256, size=(4))#generate random symbols
Kbin = np.array([int(_) for _ in ''.join(vec_dec2bin(Kdec))])
KbinRe = Kbin.reshape(-1, 4)
encoded = np.mod(np.matmul(KbinRe, G), 2)#bit after LDPC encoding


#____________________________________________________________________

encoded[0][1] = 0
print(encoded)
#______________________________________________________________________
hard_decode = np.mod(np.matmul(H, encoded[0]), 2)
print(hard_decode)
#_____________________________________________________________________

soft_decode = sumProduct(encoded[0], H, 1, maxiter=10)
print(soft_decode)
