# coding: UTF-8

import torch
import torch.nn.functional as F

class RAT():   
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def perturb(self, epsilon= 1, emb_name='bert.embeddings.word_embeddings.weight'):

        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name: 
                
                self.backup[name] = param.data.clone()  
                norm = torch.norm(param.grad)  
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm 
                    param.data += r_adv
    # restore param
    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
       
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def compute_kl_loss(self, p, q):

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')


        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

 
        
 
