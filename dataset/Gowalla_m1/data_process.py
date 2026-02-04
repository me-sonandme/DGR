import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time


def data_param_prepare():
    train_file_path='./train.txt'
    test_file_path='./test.txt'
    dataset ='Gowalla_m1'
    ii_neg_neighbor_num=30
    ii_neighbor_num=10
    lambda_1=5e-4
    lambda_2=5e-7
    
    params = {}
    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, test_file_path)
    # train_data: list of [user, item] pairs
    # train_mat: sparse adjacency matrix [n_users, n_items]
    # constraint_mat: dict with beta_uD and beta_iD for normalization
    #    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    #    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    # Training data is shuffled

    params['user_num'] = user_num
    params['item_num'] = item_num
    
    # mask matrix for testing to accelarate testing speed
    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)
    # interacted_items[u] contains all items user u has interacted with

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)
    # Test set interaction list

  
    # Compute \Omega to extend the item-item co-occurrence graph
    ii_cons_mat_path = './' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = './' + dataset + '_ii_neighbor_mat'
    
    ii_neg_neighbor_mat_path='./' + dataset + '_ii_neg_neighbor_mat'
    ii_neg_constraint_mat_path='./' + dataset + '_ii_neg_constraint_mat'
    print("********************")
    print(ii_neigh_mat_path)
    if os.path.exists(ii_neg_neighbor_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
        ii_neg_neighbor_mat=pload(ii_neg_neighbor_mat_path)
        ii_neg_constraint_mat=pload(ii_neg_constraint_mat_path)
    else:
        ii_neg_neighbor_mat,ii_neg_constraint_mat=get_ii_neg_constraint_mat(train_mat,ii_neg_neighbor_num)
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)  # Get top-k similar items
        # ii_neighbor_mat: n_items * k, indices of top-k similar items
        # ii_constraint_mat: n_items * k, normalized similarity scores
        print(ii_neg_neighbor_mat.size(),ii_neg_constraint_mat.size())
        
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)
        pstore(ii_neg_neighbor_mat,ii_neg_neighbor_mat_path)
        pstore(ii_neg_constraint_mat,ii_neg_constraint_mat_path)

    return ii_constraint_mat, ii_neighbor_mat,ii_neg_neighbor_mat,ii_neg_constraint_mat,lambda_1,lambda_2

def get_ii_neg_constraint_mat(train_mat,ii_neg_neighbor_num):
    A=train_mat.T.dot(train_mat)
    n_items=A.shape[0]
    res_mat=torch.zeros((n_items,ii_neg_neighbor_num))
    res_sim_mat=torch.zeros((n_items,ii_neg_neighbor_num))

    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1) 
    
    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  #m*m
    
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        
        non_zero_idx=torch.nonzero(row)
        idx=non_zero_idx[:,0].reshape(-1)
        #print(idx.size())
        row_nonzero=row[idx]
        
        
        idx = torch.randperm(row_nonzero.nelement())
        row_nonzero = row_nonzero[idx]
        # print(idx.size())

        if row_nonzero.size()[0] < 3*ii_neg_neighbor_num:
            res_mat[i]=torch.arange(ii_neg_neighbor_num)
            res_sim_mat[i]=torch.zeros(ii_neg_neighbor_num)
        else:
            row_sims, row_idxs = torch.topk(-1.0*row_nonzero, ii_neg_neighbor_num)
            # print("neg",row_sims)
            # print("neg",row_idxs)
            # print("nonzero",torch.nonzero(row_sims).size())
            res_mat[i]=idx[row_idxs]
            res_sim_mat[i]=-1.0*row_sims
        
    return res_mat.long(), res_sim_mat.float()
        
        
    
def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# Transpose n*m to m*n, result is m*m item-item co-occurrence matrix
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:  # Set diagonal to zero (self co-occurrence)
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)  # Row sum (item degrees in co-occurrence matrix)
    
    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  #m*m
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        # print("pos",row_sims)
        # print("pos",row_idxs)
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()

# Uses pre-split train.txt and test.txt from LightGCN
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)  # Unique users
                trainUser.extend([uid] * len(items))  # Extend with repeated user ids
                trainItem.extend(items)
                m_item = max(m_item, max(items))  # Max item id
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []   

    n_user += 1  # Add 1 to max id to get count
    m_item += 1
    filtered_lines = [str(trainUser[i])+","+str(trainItem[i])+"\n"  for i in range(len(trainUser)) ]
    train_file1="./gow_train.txt"
    with open(train_file1, 'w') as f2:
        f2.writelines(filtered_lines)
    # Open original text file for reading and processing
    print("okok")
    test_file1="./gow_test.txt"
    
    filtered_lines = [str(testUser[i])+","+str(testItem[i])+"\n" for i in range(len(testUser)) ]
    with open(test_file1, 'w') as f:
        f.writelines(filtered_lines)
    print("okok")
    
    
    time.sleep(10000)
    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
        
    

    

    
    # train_data: list of [user, item] pairs
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    # Sparse matrix n*m (not symmetric)
    
    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    # Degree-based normalization weights
    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    # print(len(train_data),len(train_data[0]),len(train_data[1]))
    # print(train_data[0],train_data[1])
    
    return train_data, test_data, train_mat, n_user, m_item, constraint_mat
def UniformSample_original_python(train_data,interacted_items,params):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    n_users=params['user_num']
    m_items= params['item_num']
    # total_start = time()
    user_num = len(train_data)  # Total number of interactions
    users = np.random.randint(0, n_users, user_num)
    allPos = interacted_items
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        # start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        # sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0,m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    #     end = time()
    #     sample_time1 += end - start
    # total = time() - total_start
    return np.array(S)

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))




if __name__ == "__main__":
   train_data, test_data, train_mat, user_num, item_num, constraint_mat = data_param_prepare()

