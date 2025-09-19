import numpy as np
from numpy import linalg as LA
from ncon import ncon

def numpy_matrices_equal(matrix1, matrix2):
    return np.array_equal(matrix1, matrix2)

n_sites=6
sI = np.eye(2)  # 单位矩阵
sz = np.array([[1, 0], [0, -1]])#Pauli Z矩阵
sx = np.array([[0, 1], [1, 0]]) #Pauli X矩阵
# 连续计算直积态的函数
def multi_kron(*matrices):
    result = matrices[0]#把result初始化为第一个矩阵
    for mat in matrices[1:]:#从第二个开始连续进行直积操作
        result = np.kron(result, mat)
    return result
# 哈密顿量H集合
H_terms = [
    -multi_kron(sz, sz, sI, sI, sI, sI),
    -multi_kron(sI, sz, sz, sI, sI, sI),
    -multi_kron(sI, sI, sz, sz, sI, sI),
    -multi_kron(sI, sI, sI, sz, sz, sI),
    -multi_kron(sI, sI, sI, sI, sz, sz),
    multi_kron(sx, sI, sI, sI, sI, sI),
    multi_kron(sI, sx, sI, sI, sI, sI),
    multi_kron(sI, sI, sx, sI, sI, sI),
    multi_kron(sI, sI, sI, sx, sI, sI),
    multi_kron(sI, sI, sI, sI, sx, sI),
    multi_kron(sI, sI, sI, sI, sI, sx)]
# 将所有项相加得到总哈密顿量
H = sum(H_terms)

def doApplyMPO(psi, L, M1, M2, R):
    """ function for applying MPO to state """

    return ncon([psi.reshape(L.shape[2], M1.shape[3], M2.shape[3], R.shape[2]), L, M1, M2, R],
                [[1, 3, 5, 7], [2, -1, 1], [2, 4, -2, 3], [4, 6, -3, 5], [6, -4, 7]]).reshape(
        L.shape[2] * M1.shape[3] * M2.shape[3] * R.shape[2]);

def eigLanczos(psivec, linFunct, functArgs, maxit=6, krydim=4):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""

    if LA.norm(psivec) == 0:
        psivec = np.random.rand(len(psivec))

    psi = np.zeros([len(psivec), krydim + 1])
    A = np.zeros([krydim, krydim])
    dval = 0

    for ik in range(maxit):

        psi[:, 0] = psivec / max(LA.norm(psivec), 1e-16)#对矢量范数归一化
        for ip in range(1, krydim + 1):#注意从ip=1循环到ip=4，最终会产生五个迭代矢量，定义最后一个psi[:,4]主要是为了后面构建A矢量而已。

            psi[:, ip] = linFunct(psi[:, ip - 1], *functArgs)#对矢量作用一次有效哈密顿量得到迭代矢量

            for ig in range(ip):#注意此为定义厄米对称矩阵的方法,得到4*4矩阵
                A[ip - 1, ig] = np.dot(psi[:, ip], psi[:, ig])#定义矩阵，每一个ip下定义A[ip,0]-A[ip-1,ip-1]，其值为(H_|psi_(ip-1)>)_conj|psi_ig>
                A[ig, ip - 1] = np.conj(A[ip - 1, ig])#定义其对称位点的矩阵元.
                # 最终的A矩阵即为在psi[:,0]...psi[:,krydim]为基矢的子空间(Krylov子空间)内写出哈密顿量的矩阵形式

            for ig in range(ip):
                psi[:, ip] = psi[:, ip] - np.dot(psi[:, ig], psi[:, ip]) * psi[:, ig]#对得到的迭代矢量进行施密特正交化
                psi[:, ip] = psi[:, ip] / max(LA.norm(psi[:, ip]), 1e-16)#对矢量归一化


        [dtemp, utemp] = LA.eigh(A)#对对称矩阵进行哈密顿矩阵对角化得到特本征值dtemp(按照升序排列)和对应的本征向量utemp
        psivec = psi[:, range(0, krydim)] @ utemp[:, 0]#把得到的迭代矢量同最小本征值对应的本征向量进行相乘得到输出

    psivec = psivec / LA.norm(psivec)
    dval = dtemp[0]

    return psivec, dval

n_sites=6#取计算自旋数目

shape=(2,2,2,2,2,2) #创建张量维度
tensor_ = np.zeros(shape)#构建全为0的初始张量
dims = tensor_.shape  # 获取张量形状
# tensor_6d[0,0,0,0,0,0]=1
# tensor_6d[1,1,1,0,0,1]=1
# tensor_6d[1,1,0,1,0,1]=1
# tensor_6d[0,0,0,1,0,1]=1
# tensor_6d[1,1,1,0,1,0]=1

tensor_ = np.random.random(shape)#随机生成张量

#进行构建MPS
def mps_decomposition(tensor_, bond_dims=None):
    dims = tensor_.shape  # 获取张量形状
    n=len(dims)
    if bond_dims is None:#设置中间的键指标
        bond_dims = [min(np.prod(dims[:i]), np.prod(dims[i:])) for i in range(1, n)]

    cores = []  # 初始化核心张量列表

    # 第一步SVD分解
    matrix = tensor_.reshape(dims[0], -1)  # 重塑为矩阵
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)  # 进行SVD分解

    bond_dim = min(bond_dims[0], s.shape[0])
    u = u[:, :bond_dim]  # 保留对应的左奇异向量
    s = s[:bond_dim]  # 保留对应的奇异值
    vh = vh[:bond_dim, :]  # 保留对应的右奇异向量

    core1 = u.reshape(1,dims[0], bond_dim)  # 重塑为核心张量
    cores.append(core1)  # 添加到核心张量列表
    matrix = np.diag(s) @ vh  # 重建剩余矩阵

    # 循环进行后续的SVD分解
    for i in range(1, n_sites-1):
        matrix = matrix.reshape(bond_dim * dims[i], -1)  # 重塑为矩阵

        ui, si, vhi = np.linalg.svd(matrix, full_matrices=False)  # 进行SVD分解
        #
        bond_dim_prev = bond_dim  # 保存之前的键维度
        bond_dim = min(bond_dims[i], si.shape[0])
        ui = ui[:, :bond_dim]  # 保留对应的左奇异向量
        si = si[:bond_dim]  # 保留对应的奇异值
        vhi = vhi[:bond_dim, :]  # 保留对应的右奇异向量

        core2 = ui.reshape(bond_dim_prev, dims[i], bond_dim)  # 重塑为核心张量
        cores.append(core2)  # 添加到核心张量列表
        matrix = np.diag(si) @ vhi  # 重建剩余矩阵

    # 处理最后一个核心张量
    core6 = matrix.reshape(bond_dim, dims[1],1)  # 重塑为最后一个核心张量
    cores.append(core6)

    return cores

mps=mps_decomposition(tensor_, bond_dims=None)#把结果给到mps当中
# print(mps)
#
#

#进行一维伊辛链哈密顿量MPO的构造。

# 定义物理维度
chid = 2
mpo=[]
# 定义泡利矩阵
sX = np.array([[0, 1], [1, 0]])
sZ = np.array([[1, 0], [0, -1]])
sI = np.array([[1, 0], [0, 1]])
# 设置模型参数
J = 1.0  # 相互作用强度
h = 1  # 横向磁场强度

# 初始化MPO张量
M = np.zeros([3, 3, chid, chid])
# 填充MPO矩阵元素
M[0, 0, :, :] = sI  # 单位算符
M[0, 1, :, :] = sZ  # σ^z算符
M[0, 2, :, :] = h * sX  # 横向磁场项
M[1, 2, :, :] = -J * sZ  # 相互作用项
M[2, 2, :, :] = sI  # 单位算符
MPO=[]

for i in range(n_sites):
    MPO.append(M)
# 定义边界向量
ML = np.array([1, 0, 0]).reshape(3, 1, 1)  # 左边界
MR = np.array([0, 0, 1]).reshape(3, 1, 1)  # 右边界


Hmatrix=ML.reshape(1,3,1,1)

for p in range(n_sites):
    Hmatrix=ncon([Hmatrix,MPO[p]],[[-1,1,-3,-5],[1,-2,-4,-6]])
    Hmatrix=Hmatrix.reshape(1,Hmatrix.shape[1],
                                    Hmatrix.shape[2]*Hmatrix.shape[3],Hmatrix.shape[4]*Hmatrix.shape[5])

Hmatrix=ncon([Hmatrix,MR.reshape(3)],[[-1,1,-2,-3],[1]]).reshape(2**n_sites,2**n_sites)
print(Hmatrix)
c=numpy_matrices_equal(H, Hmatrix)
print(c)

# n_sites=6
# sI = np.eye(2)  # 单位矩阵
# sz = np.array([[1, 0], [0, -1]])#Pauli Z矩阵
# sx = np.array([[0, 1], [1, 0]]) #Pauli X矩阵
# # 连续计算直积态的函数
# def multi_kron(*matrices):
#     result = matrices[0]#把result初始化为第一个矩阵
#     for mat in matrices[1:]:#从第二个开始连续进行直积操作
#         result = np.kron(result, mat)
#     return result
# # 哈密顿量H集合
# H_terms = [
#     -multi_kron(sz, sz, sI, sI, sI, sI),
#     -multi_kron(sI, sz, sz, sI, sI, sI),
#     -multi_kron(sI, sI, sz, sz, sI, sI),
#     -multi_kron(sI, sI, sI, sz, sz, sI),
#     -multi_kron(sI, sI, sI, sI, sz, sz),
#     multi_kron(sx, sI, sI, sI, sI, sI),
#     multi_kron(sI, sx, sI, sI, sI, sI),
#     multi_kron(sI, sI, sx, sI, sI, sI),
#     multi_kron(sI, sI, sI, sx, sI, sI),
#     multi_kron(sI, sI, sI, sI, sx, sI),
#     multi_kron(sI, sI, sI, sI, sI, sx)]
# # 将所有项相加得到总哈密顿量
# H = sum(H_terms)
# mpo=[]#初始化mpo序列
#
# bond_dims=[]
# for i in range(n_sites-1):
#     bond_dims .append (min(2 ** (2 * (i + 1)), 2 ** (2 * (n_sites - 1 - i))))#构建键指标
# # print(bond_dims[0], bond_dims[2])检验键指标是否构建正确
#
# H=H.reshape(2**2,2**(2*n_sites-2))#给哈密顿量做成
# u, s, vh = np.linalg.svd(H, full_matrices=False)  # 进行SVD分解
#
# bond_dim=min(bond_dims[0], s.shape[0])#更新键指标
# u = u[:, :bond_dim]
# s = s[:bond_dim]
# vh=vh[:bond_dim, :]
# u_=u.T#由于mpo的键指标一般占据前两个指标，因此转置后便于进行重构成为mpo
# # print(u_)
# # print(u)
# u=u_.reshape(1,bond_dim,2,2)
# mpo.append(u)
# # print(mpo[0].shape )
# H=np.diag(s) @ vh#重新形成剩余的哈密顿量
# print(H.shape)
#
# for i in range(1,n_sites-1):
#    H_=H.reshape(bond_dim*2,2**(n_sites-i-1),2,2**(n_sites-i-1))#注意对于哈密顿量张量表示各条腿编号！
#
#    H=(np.transpose(H_,(0,2,1,3))).reshape(bond_dim*4,2**(2*n_sites-2*i-2))#把需要构建的两条腿放前面，再重构为矩阵
#    u1,s1,vh1=np.linalg.svd(H, full_matrices=False) #进行svd操作
#
#    bond_dim_prev=bond_dim
#    bond_dim=min(bond_dims[i], s1.shape[0])
#    u1=u1[:,:bond_dim]
#    s1=s1[:bond_dim]
#    vh1=vh1[:bond_dim, :]#对svd得到的奇异值进行截断
#
#    u1_=u1.reshape(bond_dim_prev,2,2,bond_dim)
#    u1=np.transpose(u1_,(0,3,1,2))#重构使键指标在前两个指标
#    mpo.append(u1)#添加得到的mpo
#
#    H=(np.diag(s1) @ vh1)
#
# mpo_last=(np.diag(s1)@vh1).reshape(bond_dim,1,2,2)
#
# mpo.append(mpo_last)
#
# print(mpo[-1].shape)

def doDMRG_MPO(A, ML, M, MR, chi, numsweeps=50, dispon=2, updateon=True, maxit=6, krydim=4):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 19/1/2019
------------------------
Implementation of DMRG for a 1D chain with open boundaries, using the \
two-site update strategy. Each update is accomplished using a custom \
implementation of the Lanczos iteration to find (an approximation to) the \
ground state of the superblock Hamiltonian. Input 'A' is containing the MPS \
tensors whose length is equal to that of the 1D lattice. The Hamiltonian is \
specified by an MPO with 'ML' and 'MR' the tensors at the left and right \
boundaries, and 'M' the bulk MPO tensor. Automatically grow the MPS bond \
dimension to maximum dimension 'chi'. Outputs 'A' and 'B' are arrays of the \
MPS tensors in left and right orthogonal form respectively, while 'sWeight' \
is an array of the Schmidt coefficients across different lattice positions. \
'Ekeep' is a vector describing the energy at each update step.

Optional arguments:
`numsweeps::Integer=10`: number of DMRG sweeps
`dispon::Integer=2`: print data never [0], after each sweep [1], each step [2]
`updateon::Bool=true`: enable or disable tensor updates
`maxit::Integer=2`: number of iterations of Lanczos method for each diagonalization
`krydim::Integer=4`: maximum dimension of Krylov space in superblock diagonalization
"""

    ##### left-to-right 'warmup', put MPS in right orthogonal form

    Nsites = len(A)#定义格点数目
    L = [0 for x in range(Nsites)];
    L[0] = ML#初始化左环境张量，设置为1*1*1三阶张量
    R = [0 for x in range(Nsites)];
    R[Nsites - 1] = MR#初始化右环境张量
    for p in range(Nsites - 1):#构建正交化张量
        chid = M[p].shape[2]  # local dimension
        chil = A[p].shape[0];#注意A[0]的左键维度为1
        chir = A[p].shape[2]
        utemp, stemp, vhtemp = LA.svd(A[p].reshape(chil * chid, chir), full_matrices=False)#先把A[p]构建成矩阵再使用svd
        A[p] = utemp.reshape(chil, chid, chir)#重新形成一个三阶张量，若A[0]，则chil=1
        A[p + 1] = ncon([np.diag(stemp) @ vhtemp, A[p + 1]], [[-1, 1], [1, -2, -3]]) / LA.norm(stemp)#对得到的奇异值矩阵化再和A[p+1]缩并成新的A[P+1]
        L[p + 1] = ncon([L[p], M[p], A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])#缩并得到L[p+1]

    chil = A[Nsites - 1].shape[0];
    chir = A[Nsites - 1].shape[2]
    utemp, stemp, vhtemp = LA.svd(A[Nsites - 1].reshape(chil * chid, chir), full_matrices=False)
    A[Nsites - 1] = utemp.reshape(chil, chid, chir)#对最后右边界进行正交化
    sWeight = [0 for x in range(Nsites + 1)]
    sWeight[Nsites] = (np.diag(stemp) @ vhtemp) / LA.norm(stemp)#定义最右侧的sWeight张量
    #注意此时的A列表以及sWeight列表当中的张量的范数都不一定是1

    Ekeep = np.array([])
    B = [0 for x in range(Nsites)]
    for k in range(1, numsweeps + 2):

       #最后一次主要是为了实现正交化
        if k == numsweeps + 1:
            updateon = False
            dispon = 0

        for p in range(Nsites - 2, -1, -1):#从右向左扫描，由于一次为两个mps所以开始为n-2
            chil = A[p].shape[0];#组合张量最左侧指标长度
            chir = sWeight[p+2].shape[1]#组合张量最右侧的指标长度,注意A[p].shape[0]与sWeight[p+1].shape[1]，sWeight[p+1].shape[0]是相等的
            # print(chir)

            psiGround = ncon([A[p], A[p + 1], sWeight[p + 2]], [[-1, -2, 1], [1, -3, 2], [2, -4]]).reshape(
                chil * chid * chid * chir)#构造这两个格点的波函数
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M[p], M[p+1], R[p + 1]), maxit=maxit,
                                               krydim=krydim)#使用Lanczos方法寻找基态
                Ekeep = np.append(Ekeep, Entemp)#记录基态能量值

            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)#再次利用svd

            A[p] = utemp[:, :len(stemp)].reshape(chil, chid, len(stemp))
            sWeight[p + 1] = np.diag(stemp[:len(stemp)] / LA.norm(stemp[:len(stemp)]))#从右向左逐渐定义sWeight，该张量始终是方形的
            B[p + 1] = vhtemp[:len(stemp), :].reshape(len(stemp), chid, chir)#从右向左逐渐定义B
            R[p] = ncon([M[p+1], R[p + 1], B[p + 1], np.conj(B[p + 1])], [[-1, 2, 3, 5], [2, 1, 4], [-3, 5, 4], [-2, 3, 1]])#从右向左逐渐定义R

            if dispon == 2:
                print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))#显示此时扫描的次数，能量值等信息


        chil = A[0].shape[0];
        chir = A[0].shape[2]
        Atemp = ncon([A[0], sWeight[1]], [[-1, -2, 1], [1, -3]]).reshape(chil, chid * chir)#对左边界进行处理
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        B[0] = vhtemp.reshape(chil, chid, chir)
        sWeight[0] = utemp @ (np.diag(stemp) / LA.norm(stemp))#完全相同的处理方法得到最左侧的

        for p in range(0,Nsites - 1):#考虑从左向右再次进行优化计算

            ##### two-site update
            chil = B[p].shape[0];#注意B[p].shape[0]与sWeight[p].shape[1]，sWeight[p].shape[0]是相等的
            chir = B[p + 1].shape[2]
            psiGround = ncon([sWeight[p], B[p], B[p + 1]], [[-1, 1], [1, -2, 2], [2, -3, -4]]).reshape(
                chil * chid * chid * chir)
            if updateon:
                psiGround, Entemp = eigLanczos(psiGround, doApplyMPO, (L[p], M[p], M[p+1], R[p + 1]), maxit=maxit,
                                               krydim=krydim)
                Ekeep = np.append(Ekeep, Entemp)


            utemp, stemp, vhtemp = LA.svd(psiGround.reshape(chil * chid, chid * chir), full_matrices=False)#重新由SVD进行正交化

            A[p] = utemp[:, :len(stemp)].reshape(chil, chid, len(stemp))
            sWeight[p + 1] = np.diag(stemp[:len(stemp)] / LA.norm(stemp[:len(stemp)]))
            B[p + 1] = vhtemp[:len(stemp), :].reshape(len(stemp), chid, chir)

            ##### new block Hamiltonian
            L[p + 1] = ncon([L[p], M[p], A[p], np.conj(A[p])], [[2, 1, 4], [2, -1, 3, 5], [4, 5, -3], [1, 3, -2]])


            ##### display energy
            # if dispon == 2:
            #     print('Sweep: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        ###### right boundary tensor
        chil = B[Nsites - 1].shape[0];
        chir = B[Nsites - 1].shape[2]
        Atemp = ncon([B[Nsites - 1], sWeight[Nsites - 1]], [[1, -2, -3], [-1, 1]]).reshape(chil * chid, chir)
        utemp, stemp, vhtemp = LA.svd(Atemp, full_matrices=False)
        A[Nsites - 1] = utemp.reshape(chil, chid, chir)
        sWeight[Nsites] = (stemp / LA.norm(stemp)) * vhtemp

        # if dispon == 1:
        #     print('Sweep: %d of %d, Energy: %12.12d, Bond dim: %d' % (k, numsweeps, Ekeep[-1], chi))

    return Ekeep, A, sWeight, B
DMRG=doDMRG_MPO(mps, ML, MPO, MR, 8, numsweeps=80, dispon=2, updateon=True, maxit=6, krydim=4)

A=DMRG[1]
sWeight=DMRG[2]#将得到的最终的A与sWeight张量幅值

psi_=A[0].reshape(A[0].shape[0]*A[0].shape[1],A[0].shape[2])
for j in range(len(A)-1):

    psi_=ncon([psi_,A[j+1]],[[-1,1],[1,-2,-3]])
    psi_=psi_.reshape(psi_.shape[0]*psi_.shape[1],A[j+1].shape[2])

psi_=ncon([psi_,sWeight[-1]],[[-1,1],[1,-2]]).reshape(psi_.shape[0]*psi_.shape[1])
psi_=psi_/LA.norm(psi_)

print("使用DMRG得到的基态",psi_)
# print(DMRG[0])

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
print("使用精确对角化的基态",psi[:,0])
print(Energy[0])

def psi_product(psi1,psi2):
    if len(psi1)!=len(psi2):
        print('Psi product error: len(psi1)!=len(psi2)')#判断两个输入的波函数的长度是否一致

    else:
       product=0#初始化内积值
       for i in range(len(psi1)):
           product += (psi1[i])*(psi2[i])

    return product


product=psi_product(psi_,psi)
print("两个基态的内积为",product)












