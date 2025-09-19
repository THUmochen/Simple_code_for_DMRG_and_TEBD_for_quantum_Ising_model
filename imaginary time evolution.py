import numpy as np

from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs
from numpy import linalg as LA
from scipy.linalg import expm
import matplotlib.pyplot as plt

def psi_product(psi1,psi2):
    if len(psi1)!=len(psi2):
        print('Psi product error: len(psi1)!=len(psi2)')#判断两个输入的波函数的长度是否一致

    else:
       product=0#初始化内积值
       for i in range(len(psi1)):
           product += (psi1[i])*(psi2[i])#进行每一个位点上的数值相乘得到内积
       if abs(product)<=abs(psi1[0]/(1e5)):
           print(' psi1 is perpendicular to psi2')
       elif abs(product-1)<=1e-5:
           print(' psi1 is parallel to psi2')

    return product

def multi_kron(matrices):
    result = matrices[0]#把result初始化为第一个矩阵
    for mat in matrices[1:]:#从第二个开始连续进行直积操作
        result = np.kron(result, mat)
    return result

# matrices=[]
# matrices.append(sX)
# matrices.append(sX)
# A=multi_kron(matrices)
# print(A)


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


n_sites=6#定义自旋数目
length=2**n_sites
vector=np.zeros(length)#初始化向量值
J=1#z方向自旋相互作用强度
h=1#磁场强度
tau=0.01#更新间隔

vector=np.random.random(length)#设置随机化向量值，作为初始态演化波函数
print("初始的矢量:",vector)
vector_normalized=(vector/LA.norm(vector)).reshape(length,1)
# print(vector_normalized)
# a=psi_product(vector_normalized,vector_normalized)

sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
I=np.array([[1, 0], [0, 1]])#构造泡利矩阵
times=1000#定义循环次数

H_term=hermitian(sZ,sX,n_sites,J,h,tau)[0]
H=hermitian(sZ,sX,n_sites,J,h,tau)[1]

Hermitian=sum(H)#对每一项H_term进行相加得到总哈密顿量
# print(H[5])
psiout=vector_normalized

k=100#设置需要取的样本数目

psiout_interal=[]

E=np.zeros(times)#注意如果会出现有复数的情况，一定要记得加上 dtype=complex
for m in range(times):
    for i in range(len(H_term)):
        psiout=ncon([H_term[i],psiout],[[-1,1],[1,-2]])#将局域哈密顿量作用上去

    psiout=(psiout/LA.norm(psiout)).reshape(length)#归一化并做成向量形式
    if m%10==0:
        psiout_interal.append(psiout)

    psiout_=np.conj(psiout).reshape(length)#得到共轭向量

    E[m]=np.real(ncon([psiout_,Hermitian,psiout],[[1],[1,2],[2]]))#定义第m+1个能量值

    psiout = (psiout / LA.norm(psiout)).reshape(length,1)#重新reshape便于张量缩并
    psiout_ = np.conj(psiout).reshape(length,1)


Einternal=np.zeros(k)#初始化样本矢量
for m in range(k):
    Einternal[m]=E[10*m]#幅值

print("Einternal:",Einternal[:])
# print("能量随时间的演化",E[:])
# Max=max(E)
# Min=min(E)
# gap=Max-Min
# print("TEBD输出的能量的偏差：",gap)
psiout=psiout.reshape(length)
print("演化最后的矢量:",psiout)
Tau = np.arange(0, times*tau, tau)#构造虚时坐标


print(Tau.shape[0])

def doApplyHam(psiIn, hloc, N, usePBC):
    d = 2

    psiOut = np.zeros(psiIn.size)
    # 应用键相互作用和外场项（通过键添加的部分）
    for k in range(N - 1):

        psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                               psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                               axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)
    # 额外添加端点格点的外场项：由于键添加的外场项只给了端点0.5的系数，这里补上剩余的0.5
    # 但注意：当前hloc的外场项系数是0.5，所以总外场项需要额外添加每个端点0.5的σ_x
    if not usePBC:

        # 添加右端点格点的外场项：格点N-1
        h_right = np.kron(sI, sx)
        # gatelast=expm(- tau *h_right.reshape(d,d) ).reshape(d, d)
        psiOut += np.tensordot(h_right.reshape(d**2,d**2), psiIn.reshape(d**(N-2), d**2), axes=[[1], [1]]).transpose(1,0).reshape(d**N)
    return psiOut


"""
mainExactDiag.py
---------------------------------------------------------------------
Script file for initializing exact diagonalization using the 'eigsh' routine
for a 1D quantum system.

"""

from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer#可以用于测量代码执行时间


#设置参数
Nsites = 6  # 自旋数目选取
usePBC = False  # 是否考虑周期性边界条件
numval = 1  # 求解的本征态数目

d = 2  # 每一个自旋的维度
sx = np.array([[0, h], [h, 0]])
sz = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])#分别定义泡利矩阵
sy = np.array([[0, -1j], [1j, 0]])

hloc = (-np.kron(sz, sz)+np.kron(sx, sI)
          ).reshape(2, 2, 2, 2)#构造局域哈密顿量，其中kron为直积
# holc=expm(-tau*hloc.reshape(d**2,d**2)).reshape(2,2,2,2)
EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC
#
# print(EnExact)
def doApplyHamClosed(psiIn):#将输入波函数应用到doApplyHam函数当中
  return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')#精确对角化函数
diag_time = timer() - start_time
# psi=psi.reshape(2,2**Nsites)
# psi=psi.reshape(psi.shape[0])
EnErr = Energy[0] - EnExact  # should equal to zero
# print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e' %
#       (Nsites, diag_time, Energy[0], EnErr))
# psi=psi/LA.norm(psi)
# psi[:,1]=psi[:,1]/LA.norm(psi[:,1])
# print("使用精确对角化的基态",psi[:,0])
# print(Energy[0])

def psi_product(psi1,psi2):
    if len(psi1)!=len(psi2):
        print('Psi product error: len(psi1)!=len(psi2)')#判断两个输入的波函数的长度是否一致

    else:
       product=0#初始化内积值
       for i in range(len(psi1)):
           product += (psi1[i])*(psi2[i])

    return product

for o in range(k):

    product=abs(psi_product(psiout_interal[o],psi))
    print("两个基态的内积为",product)



# # 创建图形和坐标轴
# fig, ax = plt.subplots(figsize=(8, 6))

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
# plt.rcParams['axes.unicode_minus'] = False

# # 绘制线图
# ax.plot(Tau, E, label='sin(x)', color='red', linewidth=2)
#
# # 添加标题和标签
# ax.set_title('随虚时演化的能量值', fontsize=14)
# ax.set_xlabel('虚时时间轴', fontsize=12)
# ax.set_ylabel('能量', fontsize=12)
#
# ax.grid(True, linestyle='--', alpha=0.7)
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# 创建一些示例数据
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
# 创建图表
plt.figure(figsize=(8, 4))
plt.plot(Tau, E, label='能量变化')
plt.title('虚时演化下的态能量变化')
plt.xlabel('虚时时间轴')
plt.ylabel('能量')
plt.legend()
plt.grid(True)


# 添加水平渐近线 y=2
plt.axhline(y=-7.296230, color='r', linestyle='--', alpha=0.7, label='水平渐近线 y=-7.296230')

# 标注渐近线
plt.text(8, -7.8, 'y=-7.296230', fontsize=12, color='red',
         bbox=dict(facecolor='white', alpha=0.7))
plt.show()

