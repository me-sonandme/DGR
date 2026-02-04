import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20
import numpy as np

class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])

        # Parameterize layer coefficients for de-smoothing
        self.layer_coef_a = float(args['-layer_a']) if args.contain('-layer_a') else 0.1
        self.layer_coef_b = float(args['-layer_b']) if args.contain('-layer_b') else 0.1
        self.layer_coef_c = float(args['-layer_c']) if args.contain('-layer_c') else 0.0

        # Parameterize lambda values for loss
        self.lambda_1 = float(args['-lambda_1']) if args.contain('-lambda_1') else 1e-2
        self.lambda_2 = float(args['-lambda_2']) if args.contain('-lambda_2') else 5e-6

        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers,
                                   self.layer_coef_a, self.layer_coef_b, self.layer_coef_c)

        ii_neg_neighbor_num=10
        ii_neighbor_num=10
        train_mat=self.data.train_mat
        self.ii_neg_neighbor_mat,self.ii_neg_constraint_mat=self.get_ii_neg_constraint_mat(train_mat,ii_neg_neighbor_num)
        self.ii_neighbor_mat, self.ii_constraint_mat =self.get_ii_constraint_mat(train_mat, ii_neighbor_num)
        
    
    def get_ii_neg_constraint_mat(self,train_mat,ii_neg_neighbor_num):
        A=train_mat.T.dot(train_mat)
        n_items=A.shape[0]
        res_mat=torch.zeros((n_items,ii_neg_neighbor_num))
        res_sim_mat=torch.zeros((n_items,ii_neg_neighbor_num))

        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1) 
        
        beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        
        beta_uD[np.isinf(beta_uD)] = 0.
        beta_iD[np.isinf(beta_iD)] = 0.
        
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  #m*m
        
        for i in range(n_items):
            # Search in this row
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            
            # Get non-zero elements of this row
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
    
    def get_ii_constraint_mat(self,train_mat, num_neighbors, ii_diagonal_zero = False):
        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)	# Transpose n*m to m*n, result is m*m item-item co-occurrence matrix
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:  # Set diagonal to zero (self co-occurrence)
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1)  # Row sum (item degrees in co-occurrence matrix)
        
        beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        
        beta_uD[np.isinf(beta_uD)] = 0.
        beta_iD[np.isinf(beta_iD)] = 0.
        
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
    

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                desmoothing_loss=self.create_desmoothing_loss(user_idx,pos_idx,rec_user_emb, rec_item_emb)            
                
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size +desmoothing_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    #deloss 
    def create_desmoothing_loss(self,user_idx,pos_idx,rec_user_emb, rec_item_emb):
        loss1 = self.lambda_1 * self.cal_loss_I(user_idx, pos_idx,rec_user_emb, rec_item_emb)
        loss2 = self.lambda_2*self.cal_loss_item_neg(user_idx, pos_idx,rec_user_emb, rec_item_emb)
        
        desmoothing_loss=loss1-loss2
        return desmoothing_loss

    def cal_loss_I(self, users, pos_items,user_embs,item_embs):
        #print(self.ii_neighbor_mat.size())
        neighbor_embeds = item_embs[self.ii_neighbor_mat[pos_items]]   # Shape: batch_size * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items].cuda()   # Shape: batch_size * num_neighbors (normalized similarity scores)
        user_embeds = user_embs[users].unsqueeze(1)                   # Shape: batch_size * 1 * dim
        #print(user_embeds.device,neighbor_embeds.device,sim_scores.device)
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()  # Higher co-occurrence = higher weight in loss
        # loss=torch.mean(loss)
        loss = loss.sum()
        # print(loss.item())
        return loss
    #ddl
    def cal_loss_item_neg(self,users,pos_items,user_embs,item_embs):
 
        neighbor_embeds = item_embs[self.ii_neg_neighbor_mat[pos_items]]
        sim_scores=self.ii_neg_constraint_mat[pos_items].cuda()
        user_embeds=user_embs[users].unsqueeze(1) 
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()  # Higher co-occurrence = higher weight in loss
        # loss=torch.mean(loss)
        loss = loss.sum()
       # print(loss.item())
        return loss
    

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, layer_coef_a=0.1, layer_coef_b=0.1, layer_coef_c=0.0):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers

        # Store layer coefficients for de-smoothing
        self.layer_coef_a = layer_coef_a
        self.layer_coef_b = layer_coef_b
        self.layer_coef_c = layer_coef_c

        self.norm_adj = data.norm_adj

        self.ai=data.ai
        self.aj=data.aj
        
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        
        self.ai,self.aj=torch.tensor(self.ai,dtype=torch.float32,requires_grad=False),torch.tensor(self.aj,dtype=torch.float32,requires_grad=False)
        # print(self.ai.dtype)
        # print(self.sparse_norm_adj.dtype)
        self.ai=self.ai.to_sparse().cuda()
        self.aj=self.aj.to_sparse().cuda()
        

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        # for k in range(self.layers):
        #     ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        #     all_embeddings += [ego_embeddings]
        all_emb1= torch.sparse.mm(self.sparse_norm_adj,ego_embeddings)
        m=torch.sparse.mm(self.aj,all_emb1)
        m=torch.sparse.mm(self.ai,m)
        all_emb1=(1+self.layer_coef_a)*all_emb1-self.layer_coef_a*m

        all_embeddings.append(all_emb1)
        
        all_emb2=torch.sparse.mm(self.sparse_norm_adj, all_emb1)
        m=torch.sparse.mm(self.aj,all_emb2)
        m=torch.sparse.mm(self.ai,m)
        all_emb2=(1+self.layer_coef_b)*all_emb2-self.layer_coef_b*m

        all_embeddings.append(all_emb2)
        
        all_emb3=torch.sparse.mm(self.sparse_norm_adj, all_emb2)
        m=torch.sparse.mm(self.aj,all_emb3)
        m=torch.sparse.mm(self.ai,m)
        all_emb3=(1+self.layer_coef_c)*all_emb3-self.layer_coef_c*m

        all_embeddings.append(all_emb3)
        
        # all_emb4=torch.sparse.mm(self.sparse_norm_adj, all_emb3)
        # m=torch.sparse.mm(self.aj,all_emb4)
        # m=torch.sparse.mm(self.ai,m)
        # all_emb4=1.05*all_emb4-0.05*m
     
        # all_embeddings.append(all_emb4)        
            

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


