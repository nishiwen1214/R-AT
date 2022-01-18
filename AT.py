# coding: UTF-8

import torch
class FGM():   # Fast Gradient Method
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def perturb(self, epsilon= 1, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name:  
                self.backup[name] = param.data.clone()         
                norm = torch.norm(param.grad)  
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm  # r_adv = ϵ⋅g/||g||2
                    param.data += r_adv
                    
    # restore param
    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
 
