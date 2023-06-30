import torch
import dgl
from layers import TimeEncode
import time

class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')
        
        
    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)
        

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')
    
    def prep_input_mails(self, mfg, uni_node, inv_node, uni_edge, inv_edge, use_pinned_buffers=False):
        t_idx = 0
        t_cuda = 0
        t_re = 0
        for i, b in enumerate(mfg):
            uni_mem = self.node_memory[uni_node.long()]
            uni_mem = uni_mem.cuda()
            b.srcdata['mem'] = uni_mem[inv_node]
            uni_mem_ts = self.node_memory_ts[uni_node.long()]
            uni_mem_ts = uni_mem_ts.cuda()
            b.srcdata['mem_ts'] = uni_mem_ts[inv_node]
            uni_mem_input = self.mailbox[uni_node.long()]
            uni_mem_input = uni_mem_input.cuda()
            b.srcdata['mem_input'] = uni_mem_input[inv_node].reshape(b.srcdata['ID'].shape[0], -1)
            uni_mail_ts = self.mailbox_ts[uni_node.long()]
            uni_mail_ts = uni_mail_ts.cuda()
            b.srcdata['mail_ts'] = uni_mail_ts[inv_node]             
                
    def prep_input_mails_selection(self, mfg, uni_node):
        for i, b in enumerate(mfg):
            uni_mem = self.node_memory[uni_node.long()]
            uni_mem_ts = self.node_memory_ts[uni_node.long()]
            uni_mem_input = self.mailbox[uni_node.long()]
            uni_mail_ts = self.mailbox_ts[uni_node.long()]
        
        return uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts
        
    def reconstruct(self, mfg, uni_mem, uni_mem_ts, uni_mem_input, uni_mail_ts, inv_node):
        for i, b in enumerate(mfg):
            b.srcdata['mem'] = uni_mem[inv_node]
            b.srcdata['mem_ts'] = uni_mem_ts[inv_node]
            b.srcdata['mem_input'] = uni_mem_input[inv_node].reshape(b.srcdata['ID'].shape[0], -1)
            b.srcdata['mail_ts'] = uni_mail_ts[inv_node]
    
    def push_update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst]
            memory = memory[:num_true_src_dst]
            ts = ts[:num_true_src_dst]
            uni, inv = torch.unique(nid, sorted=False, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            memory = memory[perm]
            ts = ts[perm]
            return nid.to(self.device), memory, ts
    
    def update_memory_trans(self, nid, memory, ts):
        memory = memory.to(self.device)
        ts = ts.to(self.device)
        self.node_memory[nid.long()] = memory
        self.node_memory_ts[nid.long()] = ts
    
    def update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst]
            memory = memory[:num_true_src_dst]
            ts = ts[:num_true_src_dst]
            uni, inv = torch.unique(nid, sorted=False, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            memory = memory[perm]
            ts = ts[perm]
            memory = memory.to(self.device)
            ts = ts.to(self.device)
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts
        
    
    def push_update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, edges, uni_node, inv_node, neg_samples=1):
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                src = root_nodes[:num_true_edges]
                dst = root_nodes[num_true_edges:num_true_edges * 2]
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = ts[:num_true_edges * 2]
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
            elif self.memory_param['deliver_to'] == 'neighbors': #APAN
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[edges.long()]], dim=0) 
                mail_ts = ts[:num_true_edges * 2]
                mail_ts = torch.cat([mail_ts, mail_ts[edges.long()]], dim=0)
                nid = uni_node[inv_node]
                perm = torch.arange(inv_node.size(0), dtype=inv_node.dtype, device=inv_node.device)
                perm = inv_node.new_empty(uni_node.size(0)).scatter_(0, inv_node, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                    
        return nid.to(self.device), mail, mail_ts
    
    def update_mailbox_trans(self, nid, mail, mail_ts):
        if self.memory_param['mail_combine'] == 'last':
            mail = mail.to(self.device)
            mail_ts = mail_ts.to(self.device)
            self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
            self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
            if self.memory_param['mailbox_size'] > 1:
                if self.update_mail_pos is None:
                    self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                else:
                    self.update_mail_pos[nid.long()] = 1
    
    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            if block is not None:
                block = block.to(self.device)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                
                src = root_nodes[:num_true_edges]
                dst = root_nodes[num_true_edges:num_true_edges * 2]
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = ts[:num_true_edges * 2]
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                if self.memory_param['mail_combine'] == 'last':
                    mail = mail.to(self.device)
                    mail_ts = mail_ts.to(self.device)
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                    if self.memory_param['mailbox_size'] > 1:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = ts[:num_true_edges * 2]
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)
                nid = block.dstdata['ID']
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]
                mail = mail.to(self.device)
                mail_ts = mail_ts.to(self.device)
                self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid.long()] = 1
                
            else:
                raise NotImplementedError

    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)

class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()

