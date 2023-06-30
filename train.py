import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use')
parser.add_argument('--model_name', type=str, default='', help='name of stored model')
parser.add_argument('--eval_neg_samples', type=int, default=1, help='how many negative samples to use at inference. Note: this will change the metric of test set to AP+AUC to AP+MRR!')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import multiprocessing as mp
import threading
import queue
import torch
import time
import random
import dgl
import numpy as np
from modules import *
from sampler import *
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score

def preparation(ret_list, node_list, ts_list, node_feats, edge_feats, history, total, flag1, flag2, q):
    for i in range(total):
        if not flag1:
            mfgs, uni_node, inv_node, uni_edge, inv_edge = \
        to_dgl_blocks(ret_list[i], history)
            node_data, edge_data = prepare_input_selection(mfgs, node_feats, edge_feats, uni_node, uni_edge)
            if not flag2:
                if node_data is not None:
                    node_data = node_data.cuda()
                if edge_data is not None:
                    edge_data = edge_data.cuda()
                mfgs = feat_reconstruct(mfgs, node_data, edge_data, inv_node, inv_edge)
                node_data, edge_data = None, None
            uni_node_r, inv_node_r, edge_r = None, None, None
        elif flag1:
            mfgs, uni_node, inv_node, uni_edge, inv_edge = node_to_dgl_blocks(node_list[i], ts_list[i])
            mfgs_r, uni_node_r, inv_node_r, uni_edge_r, inv_edge_r = to_dgl_blocks(ret_list[i], history, cuda=False,  reverse=True)
            edge_r = mfgs_r[0][0].edges()[0]
            node_data, edge_data = None, None
        
        q.put((mfgs, uni_node, inv_node, uni_edge, inv_edge, uni_node_r, inv_node_r, edge_r, node_data, edge_data))


if __name__ == '__main__':
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    node_feats, edge_feats = load_feat(args.data)
    g, df = load_graph(args.data)
    print('load initial data finish.')
    sample_param, memory_param, gnn_param, train_param = parse_config(args.config)
    train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    val_edge_end = df[df['ext_roll'].gt(1)].index[0]

    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, memory_param, gnn_param, train_param, combined=combine_first).cuda()
    mailbox = MailBox(memory_param, g['indptr'].shape[0] - 1, gnn_dim_edge) if memory_param['type'] != 'none' else None
    creterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])
    print('initialize module part finish.')
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
        if mailbox is not None:
            mailbox.move_to_gpu()
    
    def get_staleness_constraint(data, batch_size):
        range_ = np.arange(0,len(data),batch_size)
        if range_[-1] != len(data):
            range_ = np.append(range_, [len(data)])
        #Calculate the staleness
        staleness = []
        for i in range(len(range_)):
            if i != len(range_)-1:
                update = 2*(range_[i+1]-range_[i])
                nodes = len(np.unique(data[range_[i]:range_[i+1]]))
                staleness.append(update-nodes)
        staleness = np.array(staleness)
        #return the upper bound for the staleness
        return max(staleness)
    
    def adaptive_split(threshold, data, group_index):
        start_list = []
        start = 0
        while start < len(data)-1:
            count_nodes = 0
            count_edges = 0
            node_set = set()
            for i in range(start, len(data)):
                count_edges += 2
                for j in data[i]:
                    if j not in node_set:
                        node_set.add(j)
                        count_nodes += 1
                if count_edges - count_nodes > threshold or i == len(data)-1:
                    end = i
                    break
            start_list.append(end)
            start = end
        start_list = [0] + start_list
        start_list[-1] = len(data)
        #generate the split
        split = []
        for i in range(len(start_list)):
            if i != len(start_list)-1:
                split.append(range(start_list[i], start_list[i+1]))
        for i in range(len(split)):
            group_index[split[i]] = i

        return group_index

    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    neg_link_sampler = NegLinkSampler(g['indptr'].shape[0] - 1)
    print('initialize sampler finish.')
    def eval(group_indices,mode='val'):
        neg_samples = 1
        model.eval()
        aps = list()
        aucs_mrrs = list()
        if mode == 'val':
            eval_df = df[train_edge_end:val_edge_end]
        elif mode == 'test':
            eval_df = df[val_edge_end:]
            neg_samples = args.eval_neg_samples
        elif mode == 'train':
            eval_df = df[:train_edge_end]
        with torch.no_grad():
            if mode == 'train':
                group_index = group_indices[0]
            if mode == 'val':
                group_index = group_indices[1]
            elif mode == 'test':
                group_index = group_indices[2]
            total_loss = 0
            for _, rows in eval_df.groupby(group_index):
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows) * neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
                if sampler is not None:
                    if 'no_neg' in sample_param and sample_param['no_neg']:
                        pos_root_end = len(rows) * 2
                        sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    else:
                        sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                if gnn_param['arch'] != 'identity':
                    mfgs, uni_node, inv_node, uni_edge, inv_edge = \
                    to_dgl_blocks(ret, sample_param['history'])
                else:
                    mfgs, uni_node, inv_node, uni_edge, inv_edge = node_to_dgl_blocks(root_nodes, ts)
                mfgs = prepare_input(mfgs, node_feats, edge_feats, uni_node, inv_node, uni_edge, inv_edge, combine_first=combine_first)
                if mailbox is not None:
                    mailbox.prep_input_mails(mfgs[0],uni_node, inv_node, uni_edge, inv_edge)
                pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
                total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
                total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
                y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
                y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
                aps.append(average_precision_score(y_true, y_pred))
                if neg_samples > 1:
                    aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
                else:
                    aucs_mrrs.append(roc_auc_score(y_true, y_pred))
                if mailbox is not None:
                    eid = rows['Unnamed: 0'].values
                    mem_edge_feats = edge_feats[eid].cuda() if edge_feats is not None else None
                    root_nodes_gpu = torch.from_numpy(root_nodes).cuda()
                    ts_gpu = torch.from_numpy(ts).cuda()
                    block = None
                    if memory_param['deliver_to'] == 'neighbors':
                        block,_,_,_,_ = to_dgl_blocks(ret, sample_param['history'], reverse=True)
                        block = block[0][0]
                    mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, block, neg_samples=neg_samples)
                    mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes, model.memory_updater.last_updated_ts, neg_samples=neg_samples)
            if mode == 'val':
                val_losses.append(float(total_loss))
        ap = float(torch.tensor(aps).mean())
        if neg_samples > 1:
            auc_mrr = float(torch.cat(aucs_mrrs).mean())
        else:
            auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        return ap, auc_mrr
    
    
    if not os.path.isdir('models'):
        os.mkdir('models')
    if args.model_name == '':
        path_saver = 'models/{}_{}.pkl'.format(args.data, time.time())
    else:
        path_saver = 'models/{}.pkl'.format(args.model_name)
    best_ap = 0
    best_e = 0
    val_losses = list()
    group_indices = []
    #get train split
    train_index = np.array(df[:train_edge_end].index)
    train_data = np.stack([df[:train_edge_end].src, df[:train_edge_end].dst]).T
    threshold = get_staleness_constraint(train_data, train_param['batch_size'])
    train_group_index = adaptive_split(threshold, train_data, train_index)
    group_indices.append(train_group_index)
    #get val split
    val_index = np.array(df[train_edge_end:val_edge_end].index)
    val_data = np.stack([df[train_edge_end:val_edge_end].src, df[train_edge_end:val_edge_end].dst]).T
    threshold = get_staleness_constraint(val_data, train_param['batch_size'])
    val_group_index = adaptive_split(threshold, val_data, val_index)
    val_group_index += train_group_index[-1]
    group_indices.append(val_group_index)
    #get test split
    test_index = np.array(df[val_edge_end:].index)
    test_data = np.stack([df[val_edge_end:].src,df[val_edge_end:].dst]).T
    threshold = get_staleness_constraint(test_data, train_param['batch_size'])
    test_group_index = adaptive_split(threshold, test_data, test_index)
    test_group_index += val_group_index[-1]
    group_indices.append(test_group_index)
    
    print('start training.') 
    q = queue.Queue(maxsize=1)
    flag1 = False
    flag2 = False
    if mailbox is None and gnn_param['arch'] == 'transformer_attention':
        flag2 = True
    for e in range(train_param['epoch']):
        print('Epoch {:d}:'.format(e))
        ret_list = []
        neg_list = []
        node_list = []
        ts_list = []
        time_sample = 0
        time_prep = 0
        time_tot = 0
        total_loss = 0
        # training
        model.train()
        if sampler is not None:
            sampler.reset()
        if mailbox is not None:
            mailbox.reset()
            model.memory_updater.last_updated_nid = None
        #sample for all
        t_start = time.time()
        for _, rows in df[:train_edge_end].groupby(train_group_index):
            t0 = time.time()
            neg = neg_link_sampler.sample(len(rows))
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg]).astype(np.int32)
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            t1 = time.time()
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = root_nodes.shape[0] * 2 // 3
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                    ret = sampler.get_ret()
                    t2 = time.time()
                    ret_list.append(ret)
                    node_list.append(root_nodes)
                    ts_list.append(ts)
                    flag1 = True
                    time_sample += t2 - t1
                else:
                    sampler.sample(root_nodes, ts)
                    ret = sampler.get_ret()
                    t2 = time.time()
                    ret_list.append(ret)
                    node_list.append(root_nodes)
                    ts_list.append(ts)
                    time_sample += t2 - t1
            else:
                node_list.append(root_nodes)
                ts_list.append(ts)
                flag1 = True
        
        if sampler is not None:
            total = len(ret_list)
        else:
            total = len(node_list)
        t_prep_s = time.time()
        print('sample & finish, start the subthread.')
        prep_thread = threading.Thread(target = preparation, args = \
        (ret_list, node_list, ts_list, node_feats, edge_feats, sample_param['history'], total, flag1, flag2, q,))
        prep_thread.start()
        t_end_s = time.time()
        time_prep += t_prep_s - t_start
        time_tot += t_end_s - t_start
        
        #start the main computation
        count = 0
        for _, rows in df[:train_edge_end].groupby(train_group_index):
            t0 = time.time()
            root_nodes = node_list[count]
            ts = ts_list[count]
            count += 1
            #get_data
            mfgs, uni_node, inv_node, uni_edge, inv_edge, uni_node_r, inv_node_r, edge_r, node_data, edge_data = q.get()
            #refresh uni_mem
            if mailbox is not None:
                uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts = \
                mailbox.prep_input_mails_selection(mfgs[0], uni_node)
            # to_cuda 
            if flag2:
                if node_data is not None:
                    node_data = node_data.cuda()
                if edge_data is not None:
                    edge_data = edge_data.cuda()
            if mailbox is not None: 
                uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts = \
                uni_mem.cuda(), uni_mem_ts.cuda(), uni_mem_input.cuda(), uni_mail_ts.cuda()
            t3 = time.time()
            #reconstruct input
            mfgs = feat_reconstruct(mfgs, node_data, edge_data, inv_node, inv_edge)
            if mailbox is not None:
                mailbox.reconstruct(mfgs[0], uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts, inv_node)
            t1 = time.time()
            time_prep += t1-t0
            #start pipelining
            optimizer.zero_grad()
            pred_pos, pred_neg = model(mfgs)
            loss = creterion(pred_pos, torch.ones_like(pred_pos))
            loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * train_param['batch_size']
            loss.backward()
            optimizer.step()
            del mfgs[0]
            t2 = time.time()
            if mailbox is not None:      
                mem_edge_feats = edge_feats[rows['Unnamed: 0'].values].cuda()
                root_nodes_gpu = torch.from_numpy(root_nodes).cuda()
                ts_gpu = torch.from_numpy(ts).cuda()
                #push update to GPU
                mail_nid, mail, mail_ts = mailbox.push_update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, ts_gpu, mem_edge_feats, edge_r, uni_node_r, inv_node_r)
                mem_nid, mem, mem_ts = mailbox.push_update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, root_nodes_gpu, model.memory_updater.last_updated_ts)
                #transfer the update result back to CPU
                mailbox.update_mailbox_trans(mail_nid, mail, mail_ts)
                mailbox.update_memory_trans(mem_nid, mem, mem_ts)
            t3 = time.time()
            time_prep += t3-t2
            time_tot += t3-t0
        prep_thread.join()
        print('\ttotal time:{:.2f}s prep time:{:.2f}s sample time:{:.2f}s'.format(time_tot, time_prep, time_sample))
        ap, auc = eval(group_indices, 'val')
        if e > 2 and ap > best_ap:
            best_e = e
            best_ap = ap
            torch.save(model.state_dict(), path_saver)
        print('\ttrain loss:{:.4f}  val ap:{:4f}  val auc:{:4f}'.format(total_loss, ap, auc))
        
    print('Loading model at epoch {}...'.format(best_e))
    model.load_state_dict(torch.load(path_saver))
    model.eval()
    if sampler is not None:
        sampler.reset()
    if mailbox is not None:
        mailbox.reset()
        model.memory_updater.last_updated_nid = None
        eval(group_indices, 'train')
        eval(group_indices, 'val')
    ap, auc = eval(group_indices, 'test')
    if args.eval_neg_samples > 1:
        print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, auc))
    else:
        print('\ttest AP:{:4f}  test AUC:{:4f}'.format(ap, auc))
