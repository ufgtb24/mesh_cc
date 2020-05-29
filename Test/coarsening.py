import numpy as np
import scipy.sparse



def coarsen(A,levels, biased):
    """
    Coarsen a graph, represented by its adjacency matrix A, at multiple
    levels.
    A: 按 id 升序排列的 adjacency matrix
    levels: 压缩 2**levels 倍
    """
    # graph: graph[i] 的 index 代表本层cluster 的 id(形成的顺序)
    # e.g.  graph[i][0,n]代表最先形成的cluster到第n个形成的cluster之间的连接
    # parents:下层id到上层id之间的映射。id是本层cluster形成的顺序

    A_out, parents = metis(A, levels,biased) # 3 graphs   2 parents
    # 根据最顶层id升序，返回自底向上的id二叉树，二叉树的结构定义了层间连接关系
    perms = compute_perm(parents)  # 3 perms
    perm_in=np.array(perms[0])

    # if not self_connections:
    #     A_out = A_out.tocoo()
    #     A_out.setdiag(0)
    


    return perm_in, A_out

def newadj(adj_path):
    path=adj_path.split('.')[0]
    new_path=path+'1.txt'
    fnew=open(new_path,'w')
    with open(adj_path)as f:
        G=f.readlines()
        for line in G[1:]:
            fnew.write(line)
    fnew.close()
    return new_path
    
    
def multi_coarsen(adj, coarsen_times, coarsen_level):
    '''

    :param adj_path:
    :param coarsen_times:
    :param coarsen_level:
    :return perms: [np.array([size_to_sample_from])]*coarsen_times
    :return adjs: [np.array([pt_num,ADJ_K])]*coarsen_times

    '''
    # in graph mode ,adj_lenK matters in placeholder, but the first adj is not related to adj_len,
    # which is only determined by input data
    # newpath=newadj(adj_path)
    
    # adj=np.concatenate([adj,np.zeros([adj.shape[0],adj_len-adj.shape[1]],dtype=np.int32)],axis=1)
    perms = []
    adjs = []
    adjs.append(adj) # adj 比 perm  多一个
    for i in range(coarsen_times):
        if i==0:
            
            # 可以兼容adj截断的状况，不会越界，但是不再是对称矩阵
            A_in = adj_to_A(adj)
            biased=False
            # is_symm,sub=is_Symm(A_in)
            # r,c,v=scipy.sparse.find(sub)
            # rcv=np.stack([r,c,v],axis=-1)
            # if not is_symm:
            #     print('  not symm item ')
            #     print(r)
            #     print(c)
        else:
            A_in=A_out
            biased=True
        # is_symm, sub = is_Symm(A_in)
        perm_in, A_out = coarsen(A_in, coarsen_level,biased)
        perms.append(perm_in)
        A_adj=A_out.copy()
        adj = A_to_adj(A_adj)
        adjs.append(adj)
    return perms+adjs
 
def is_Symm(W):
    return (abs(W - W.T) > 0).nnz==0, abs(W - W.T)

def metis(W, levels, biased):
    """
    Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    NOTE
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """
    # print(is_Symm(W))
    N, N = W.shape

    if biased:
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)  # 按照节点的度来确定处理顺序
    else:
        rid = np.random.permutation(range(N))
        # rid = np.arange(N)  # 调试顺序
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    
    # graphs = []
    # graphs.append(W)

    for l in range(levels):

        #count += 1

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree            # graclus weights [N]
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # check=is_Symm(W)
        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        # column-major order  row_idx  col_idx   val_idx
        # 因为W为邻接矩阵，是对称的，所以将 col_idx作为row_idx，就变成了row-major order
        cc, rr,  vv = scipy.sparse.find(W)
        # 两个顺序：1.本层点的生成顺序，即本层id； 2.基于本层id 的 degree 顺序。二者共同决定下一层 id
        # 按照本层 id 排序。在这个基准上使用上一轮得到的degree increased rid 进行索引，
        # 先聚合 degree 小的点
        # perm = np.argsort(idx_row)
        # rr = idx_row[perm]
        # cc = idx_col[perm]
        # vv = val[perm]
        # 为每个不孤立的点分配cluster, 一共有 len(rid) 个点，len(cluster_id)=len(rid)
        # 每个 vertix 的起点对应的 cluster id , 由该 vertix 被 cluster 的优先级
        # 决定，没啥意义，单纯的id
        cluster_id = metis_one_level(rr,cc,vv,rid,weights)  # rr is ordered
        parents.append(cluster_id)

        # TO DO
        # COMPUTE THE SIZE OF THE SUPERNODES AND THEIR DEGREE
        #supernode_size = full(   sparse(cluster_id,  ones(N,1) , supernode_size )     )
        #print(cluster_id)
        #print(supernode_size)
        #nd_sz{count+1}=supernode_size;

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        # 每个起点 vertix 所属的 cluster id , 由该 vertix 被 cluster 的优先级决定
        nrr = cluster_id[rr] # [num of vertix]
        # 求每个终点 vertix 对应的 cluster id , 由该 vertix 被 cluster 的优先级决定
        ncc = cluster_id[cc] #[num of vertix]
        nvv = vv  # 每个 cluster 的 weight
        Nnew = cluster_id.max() + 1  # 一共多少 cluster
        # CSR is more appropriate: row,val pairs appear multiple times
        # cluster 的两两组合 作为新的 pair , 创建新的 邻接矩阵
        # 有层间父子id映射 claster_id，和父层的拓扑关系 W, W中 weight越大表示节点连接关系的越强，能够
        # 指导下一次cluster
        # scipy adds the values of the duplicate entries:  merge weights of cluster.
        # 本操作会产生自环，合并的新节点带有自环，剩下的孤立点没有自环，下次优先合并没有自环的
        # W 的 index 代表new cluster 的 id(形成的优先性)
        # e.g.  W[0,n]代表最先形成的cluster到第n个形成的cluster之间的连接
        
        
        W = scipy.sparse.coo_matrix((nvv,(nrr,ncc)), shape=(Nnew,Nnew))
        W.eliminate_zeros() # 稀疏
        
        assert is_Symm(W)
        # Add new graph to the list of all coarsened graphs
        # graphs.append(W)

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS) 忽略自环，weight变小，该点更容易被团结。
        # 但是如果该点已经是团结过好几次的了，那么应该减小它被继续团结的可能，否则会产生吸收黑洞，
        # 所以不能忽略自环
        degree = W.sum(axis=0)

        # degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        # 根据 degree accend 将cluster id 排序。 下一层从degree小的开始 cluster 开始后续 cluster
        # 目的在于先解决孤立点
        rid = np.argsort(ss) #
    # graphs 比 parents 多一层
    # graphs: finest to coarsest  包含自环,权值可以大于1，代表连接强弱
    return W, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr,cc,vv,rid,weights):
    # 只有起点rr排好序(本层cluster的生成顺序)，终点cc是随机的

    nnz = rr.shape[0] # 所有pair的数量
    N = rr[nnz-1] + 1 # 点的数量

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0
    rc=np.stack([rr,cc],axis=-1)
    for ii in range(nnz):
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1
        rowlength[count] = rowlength[count] + 1
    
    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                    if weights[tid]==0:
                        print('zeros')
                        
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid
            # tid 第一层是随机的，以后每层是 degree 升序
            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id

def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    产生底层根据顶层排序，并加入fake_nodes后的排序，最底层会用于构建二叉树
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last))) # rank the cluster id of the last layer
        #只有最后一层需要排序 indices=[0,1,2]

    for parent in parents[::-1]:
        # from the coarsest level
        #print('parent: {}'.format(parent))

        # Fake nodes go after real ones. len(parent) is the number of real node in this layer
        # add new id for fake nodes of this layer
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            
            # 每个父节点对应一个 indices_node, 索引上一层中的子节点
            # index of where condition is true
            indices_node = list((np.where(parent == i)[0]).astype(np.int32))
            
            assert 0 <= len(indices_node) <= 2
            #print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singelton.
            if len(indices_node) is 1:
                # 本层是独生子
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
                #print('new singelton: {}'.format(indices_node))
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) is 0:
                # parent是fake point
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
                #print('singelton childrens: {}'.format(indices_node))

            indices_layer.extend(indices_node) # 每次加两个元素：上一层的两个索引
        # 将根据indices[-1](coarser layer)节点顺序推导出的 indices_layer(finer layer)
        # 放入 indices。 因此 indices 中包含的由 coarser 到 finer 的层节点索引
        indices.append(indices_layer)

    # Sanity checks.
    for i,indices_layer in enumerate(indices):
        M = M_last*2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices_layer) == M
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1] # finest to coarsest

assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
        == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])

def perm_data(x, indices):
    """
    排列成待卷积的序列,并插入 fake nodes
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    new x can be used for pooling
    :return xnew
    """
    if indices is None:
        return x

    M,channel = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((Mnew,channel))
    for i,j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < M:
            xnew[i] = x[j]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
        else:
            xnew[i] = np.zeros(channel)
    return xnew

def perm_adjacency(A, indices):
    """
    其他功能实现了从粗到细按照父子关系排列完各层的id，使得最粗层id为升序，
    本函数将原本根据本层id索引的A转换成满足上层id为顺序的A
    indices 是为了组成使最顶层id为升序的二叉树，本层的id的顺序
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    new A can be used to following convolution and pooling
    """
    if indices is None:
        return A

    M, M = A.shape
    # 45 03 21    if 345 is fake, M=3
    # 0  1  2
    # indices is one of the above two lines
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    if Mnew > M:
        # 将[M,M]补0得到[Mnew,Mnew]
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])

    # Permute the rows and the columns.
    # e.g. 254|301
    #      012
    perm = np.argsort(indices)
    # row,col= M.nonzero() 两个array表示非零值的索引，按照row由小到大排序，
    # 相当于 sparse.find(W)+排序
    # 卷积要用到并行，需要将原id按 450321 排列，如下操作
    # e.g. 把id 为0的点从矩阵的 index 0 处移动到 index 2处,这是对adj的操作，
    # 还应该有对x的对应交换操作。交换完成后可以进行 两两 max 的 pooling 操作
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    # assert np.abs(A - A.T).mean() < 1e-9
    assert type(A) is scipy.sparse.coo.coo_matrix
    return A


def adj_to_A(adj):
    '''
    in np ,used at beginging for once
    :param adj: num_points, K
    :return:
    '''
    num_points, K=adj.shape
    idx=np.arange(num_points)
    idx = np.reshape(idx, [-1, 1])    # Convert to a n x 1 matrix.
    idx = np.tile(idx, [1, K])  # [pt_num,K] Create multiple columns, each column has one number repeats repTime
    x = np.reshape(idx, [-1]) # [pt_num*K] 0000  1111 2222 3333 4444 从0开始 []
    y =np.reshape(adj, [-1]) # [pt_num * K]
    mask=np.where(y!=0)[0]
    y=y[mask]-1
    x=x[mask]
    v=np.ones_like(mask)
    A = scipy.sparse.coo_matrix((v, (x, y)), shape=(num_points, num_points))
    # A=A.tocsr()
    # A.setdiag(0)
    A.eliminate_zeros()  # 稀疏
    s=is_Symm(A)
    return A


def A_to_adj(A):
    '''
    in np, used after each time of coarsening when new A is created
    in coarsen, A id is begin from 0, while in conv, adj id is begin from 1
    :return: num_points, K
    '''
    A.setdiag(0)
    cc,rr,  val = scipy.sparse.find(A)
    # 发现 A 不对称
    # perm = np.argsort(rr)
    # rr = rr[perm]
    # cc = cc[perm]
    N,N=A.shape
    K = np.max((A != 0).sum(axis=0))

    pair_num=rr.shape[0]
    adj = np.zeros([N,K], np.int32)
    cur_row = rr[0]
    cur_col=0
    for i in range(pair_num):
        if rr[i]>cur_row:
            adj[cur_row,cur_col:]=0
            cur_row=rr[i]
            cur_col=0
        adj[rr[i],cur_col]=cc[i]+1
        cur_col+=1
    return adj


def normalize(world_coord,part_id):
    '''
    
    :param world_coord:
    :param vtype: 0DL   1DR   2UL   3 UR
    :return:
    '''
    center = np.mean(world_coord, axis=0)

    local_coord = world_coord - center
    if part_id == 1:
        local_coord = local_coord * [-1, 1, 1]
    elif part_id == 2:
        local_coord = local_coord * [1, -1, 1]
    elif part_id == 3:
        local_coord = local_coord * [-1, -1, 1]
    return [local_coord.astype(np.float32),center]

def ivs_normalize(local_coord,center,part_id):
    if part_id == 1:
        local_coord = local_coord * np.array([-1, 1, 1],dtype=np.float32)
    elif part_id == 2:
        local_coord = local_coord * np.array([1, -1, 1],dtype=np.float32)
    elif part_id == 3:
        local_coord = local_coord * np.array([-1, -1, 1],dtype=np.float32)
    world_coord = local_coord + center
    return [world_coord.astype(np.float32)]
    

