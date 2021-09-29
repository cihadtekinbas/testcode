#case 1
from scipy import sparse
from scipy import sparse
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import hstack
from scipy.sparse import vstack

trainA = sparse.load_npz("data\Aratings-transform_user.npz")
testB = sparse.load_npz("data\Bratings-transform_user.npz")
trainC=sparse.load_npz("data\Cratings-transform_user.npz")
testD=sparse.load_npz("data\Dratings-transform_user.npz")


print(trainA.shape)
print(testB.shape)
testB.resize(251457,4710)
print(trainC.shape)
print(testD.shape)

testDcoo=testD.tocoo()

dictt={}
for i in range(0,len(testDcoo.row)):
    x=testDcoo.row[i]
    if x not in dictt:
        dictt[x]=[]
    dictt[x].append(testDcoo.data[i])

data=[]
for i in dictt:
    s=0
    for j in dictt[i] :
        s=s+j
    t=s/len(dictt[i])
    for z in range(len(dictt[i])):
        data.append(t)
data=np.asarray(data)
import numpy as np
from scipy.sparse import coo_matrix
coo = coo_matrix((data, (testDcoo.row, testDcoo.col)))
print(coo.shape)
coocsr=coo.tocsr()

user_number=10000

A_B=hstack((trainA, testB))
print("buradayım")
print(trainC.shape,coocsr.shape)
numbers=[]
for i in range(user_number):
    numbers.append(i)


C_D=hstack((trainC[numbers], coocsr[numbers]))
print(A_B.shape)
print(C_D.shape)
C_D.resize(user_number,23513)
A_B_C_D=vstack((A_B, C_D))
print(A_B_C_D.shape)


from scipy.linalg import sqrtm
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
# U, s, V=np.linalg.svd(utilMat, full_matrices=False)
u, s, vt = svds(A_B_C_D, k=13512) 

s=np.diag(s)
k=13512
s=s[0:k,0:k]
u=u[:,0:k]
vt=vt[0:k,:]

s_root=np.sqrt(s)

Usk=np.dot(u,s_root)
skV=np.dot(s_root,vt)
UsV = np.dot(Usk, skV)

UsV = UsV
print("svd done")

recommend={}
for i in C_D.row:
    if i not in recommend:
        recommend[i]=[]
for i in recommend:
    recommend[i]=UsV[251457+i,18803:]
    
recommend={}
f=open("recommend","w")
f.write(str(recommend))
f.close()
# keeplist={}
# for i in range(len(C_D.row)):
#     if(C_D.col[i]>18803):
#         if C_D.row[i] not in keeplist:
#             keeplist[C_D.row[i]]=[]
#         keeplist[C_D.row[i]].append(C_D.col[i])
        
# for i in range(len(recommend)):
#     if i in recommend:

#         ind = np.argpartition(recommend[i], -10)[-10:]
#         if i in keeplist:
#             for j in ind:
#                 if j in keeplist[i]:
#                     print(j,keeplist[i])