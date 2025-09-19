import numpy as np

from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs
from numpy import linalg as LA
from scipy.linalg import expm
# from typing import Optional

n_sites=6#定义自旋数目
# length=2**n_sites
# vector=np.zeros(length)#初始化向量值

shape=(2,2,2,2,2,2) #创建六阶张量
tensor_6d = np.random.random(shape)#构建全为0的初始张量
# print("张量形状:", tensor_6d.shape)#打印张量的形状
# tensor_6d[0,0,0,0,0,0]=1#设定初始处于所有自旋都朝上的状态
# # tensor_6d[0,0,0,1,0,1]=1
dims = tensor_6d.shape  # 获取张量形状

J=1
h=1

sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
I=np.array([[1, 0], [0, 1]])#构造泡利矩阵
hamZ = ((-J*np.kron(sZ, sZ))).reshape(2, 2, 2, 2)#构建local Hamiltonian
hamX = (h*sX).reshape(2, 2)#每个格点上的\sigma x作用
# print(sX)
# print(hamZ)
d=2
tau=0.0002#更新时间间隔
times=1000#设置迭代次数

gateZ = expm(-1j* tau * hamZ.reshape(d**2, d**2)).reshape(d, d, d, d)
gateX = expm(-1j* tau * hamX.reshape(d,d)).reshape(d, d)#对于相互作用取e指数再重构成为四阶张量
# print(gateZ)

#进行构建MPS
def mps_decomposition(tensor_6d, bond_dims=None):#函数语法学习？

    if bond_dims is None:
        bond_dims = [min(np.prod(dims[:i]), np.prod(dims[i:])) for i in range(1, 6) ]#.prod语法学习？
#设置键指标，根据SVD算法，选取两边的指标最小为键指标

    cores = []#初始化核心张量列表
    matrix = tensor_6d.reshape(dims[0], -1)#把张量第一个指标作为矩阵行指标，其余指标乘积作为列指标。
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)#进行SVD分解
    bond_dim = min(bond_dims[0], s.shape[0])#判断是否这里的奇异值矩阵是否类似满秩的形式
    u=u[:,:bond_dim ]#保留对应s奇异值对应的那些类似的本征列矢
    s=s[:bond_dim]
    vh=vh[:bond_dim,:]#保留对应s奇异值对应的那些类似的本征横矢
    core1=u.reshape(1,dims[0], bond_dim)
    cores.append(core1)#把u添加到核心张量列表当中
    matrix = np.diag(s) @ vh#把s形成对角阵与vh直接矩阵相乘
    # print(cores[0])

    for i in range(1,5):
       matrix = matrix.reshape(bond_dim*dims[i],-1)#注意这里此时左端等距张量有两条自由腿，一个键张量
       ui, si, vhi = np.linalg.svd(matrix, full_matrices=False)
       bond_dim_=bond_dim
       bond_dim = min(bond_dims[i], si.shape[0])
       ui=ui[:,:bond_dim ]
       si=si[:bond_dim]
       vhi=vhi[:bond_dim,:]
       core2=ui.reshape(bond_dim_,dims[i],bond_dim)
       cores.append(core2)#完全相同的构造方法
       matrix = np.diag(si) @ vhi

    core6 = matrix.reshape(bond_dim, dims[5],1)#最后一个是一个矩阵，和上面的三阶张量不同，无法放入循环，手动加入。
    cores.append(core6)
    return cores

mps=mps_decomposition(tensor_6d, bond_dims=None)#把结果给到mps当中


# print(mps[3])
# print(mps[0].shape[2])

def TEBD(mps, gate1, gate2,times=100):

    length = len(mps)#获取mps数目
    # # print(k)
    # psiin=mps[0]#初始化psiin
    #
    # for j in range(1,k):
    #
    #     psiin=psiin.reshape(1,2**j,mps[j-1].shape[2])#把psiin重构成为三条腿的形式
    #     psiin = ncon([psiin,mps[j]],[[-1,-2,1],[1,-3,-4]])#对mps进行缩并
    #
    # psiout=psiin.reshape(shape)
    #
    # for i in range(times):
    #
    #     for j  in range(0,k-1):
    #          psiout_=psiout.reshape(2**j,2,2,2**(k-j-2))#对psiout重构，便于缩并
    #          psiout=ncon([psiout_,gate1],[[-1,1,2,-4],[-2,-3,1,2]])#把第一个量子门作用上去
    #          psiout=psiout.reshape(shape)
    #
    #     for m in range(0,k):
    #         psiout_=psiout.reshape(2**m,2,2**(k-m-1))
    #         psiout=ncon([psiout_,gate2],[[-1,1,-3],[-2,1]])#把第二个量子门作用上去
    #         psiout = psiout.reshape(shape)
    #
    # psiout=psiout.reshape(2**k)

    psiout=[]#定义输出波函数列表
    for j in range(times):

        for m in range(length-1):

            bond_dim=min(mps[m].shape[2],mps[m+1].shape[0])
            mps_togeter=ncon([mps[m],mps[m+1],gate1],[[-1,1,2],[2,3,-4],[-2,-3,1,3]])#缩并产生两个自旋的mps，并作用上量子门
            mps_change=mps_togeter.reshape(mps_togeter.shape[0]*mps_togeter.shape[1],
                                           mps_togeter.shape[2]*mps_togeter.shape[3])
            mps_m,s,mps_m1=np.linalg.svd(mps_change ,full_matrices=False)#进行svd分解

            bond_dim=min(bond_dim,s.shape[0])
            mps_m=mps_m[:,:bond_dim]
            s=s[:bond_dim]
            mps_m1=mps_m1[:bond_dim,:]#重新构建mps[m]和mps[m+1]
            mps_m1=np.diag(s) @ mps_m1
            mps_m=mps_m.reshape(mps_togeter.shape[0],mps_togeter.shape[1],bond_dim)
            mps_m1=mps_m1.reshape(bond_dim,mps_togeter.shape[2],mps_togeter.shape[3])#注意要把两个更新的矩阵重新转换为mps的三阶形式

            mps[m]=mps_m
            mps[m+1]=mps_m1#重新幅值mps[m]和mps[m+1]

        for n in range(length):

            mps_together = ncon([mps[n], gate2],
                               [[-1,1,-3],  [-2,1]])  # 缩并作用上量子门
            mps[n] = mps_together

        psiinteral=mps[0]#初始化psiinteral为mps[0]
        # print(psiinteral.shape)
        for l in range(length-1):

            psiinteral=psiinteral.reshape(2**(l+1),mps[l].shape[2])
            psiinteral=ncon([psiinteral,mps[l+1]],[[-1,1],[1,-2,-3]])#连续缩并上mps

        psiinteral=psiinteral.reshape(2**(length))/LA.norm(psiinteral)#形成一个输出矢量
        psiout.append(psiinteral)#将中间演化波函数加入到列表当中

    return psiout

def multi_kron(matrices):
    result = matrices[0]#把result初始化为第一个矩阵
    for mat in matrices[1:]:#从第二个开始连续进行直积操作
        result = np.kron(result, mat)
    return result

def hermitian(term1,term2,sites,J,h,tau):

    I = np.array([[1, 0], [0, 1]])#定义单位矩阵，适用于两自由度的子体系
    H_term=[]#初始化局域哈密顿量列表
    matrices=[]#初始化用于做直积的矩阵列表
    H=[]
    for i in range(sites-1):

        # if i==0:#构造首个用于直积的矩阵列表
        #     matrices.append(term1)
        #     matrices.append(term1)
        #     for l in range(sites-2):
        #         matrices.append(I)
        #     gate=expm(-1* tau * (-J*multi_kron(matrices))) #构造量子门
        #     H_term.append(gate)

        for k in range(i):#先加入前i个I矩阵
            matrices.append(I)

        matrices.append(term1)#把两个量子门加入其中
        matrices.append(term1)

        for l in range(sites-2-i):
            matrices.append(I)#继续加入后面的单位矩阵

        H.append(-J*multi_kron(matrices))
        gate=expm(-1* tau * (-J*multi_kron(matrices))).reshape(2**(sites),2**(sites))#构造量子门
        H_term.append(gate)#增加局域哈密顿量至列表当中
        matrices=[]#重新更新matrices列表

    for m in range(sites):
        for k in range(m):#先加入前i个I矩阵
            matrices.append(I)

        matrices.append(term2)#把量子门加入其中
        for l in range(sites - 1 - m):
            matrices.append(I)  # 继续加入后面的单位矩阵

        H.append(h*multi_kron(matrices))
        gate=expm(-1* tau * (h*multi_kron(matrices))).reshape(2**(sites),2**(sites))#构造量子门
        H_term.append(gate)#增加局域哈密顿量至列表当中
        matrices = []  # 重新更新matrices列表

    return H_term,H

Hmatrix=hermitian(sZ,sX,n_sites,J,h,tau)[1]
# print(Hmatrix)
Hreal=sum(Hmatrix)#得到哈密顿量


E=np.zeros(times)#初始化能量列表
psiout = TEBD(mps, gateZ, gateX, times)#得到演化波函数

for k in range(1,times+1):
    psioutconj=np.conj(psiout[k-1])#对输出波函数进行取复共轭
    E[k-1]=np.real((ncon([psioutconj,Hreal,psiout[k-1]],[[1],[1,2],[2]])))#求能量平均值

# print(psiout)
print(len(psiout))
print("得到的演化能量",E[:])
Emax=max(E)
Emin=min(E)
E_=Emax-Emin
print("能量峰谷的差：",E_)



