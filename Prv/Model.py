import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Utils.Utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class ModalGatingNetwork(nn.Module):
    """
    Attention-based gating: sinh ra trọng số α riêng cho mỗi modal
    dựa trên embedding hiện tại của từng node (user/item).

    Input:  h ∈ R^{N x latdim}
    Output: α ∈ R^{N x num_modals}  (softmax theo dim=-1, tổng = 1)
    """
    def __init__(self, latdim, num_modals, hidden_dim=32):
        super(ModalGatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(latdim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_modals),
        )

    def forward(self, h):
        # h: (N, latdim) → logits: (N, num_modals) → weights: (N, num_modals)
        logits = self.gate(h)
        weights = F.softmax(logits, dim=-1)
        return weights


class Model(nn.Module):
    def __init__(self, image_embedding, text_embedding, audio_embedding=None):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge(args.keepRate)

        if args.trans == 1:
            self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
        elif args.trans == 0:
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
        else:
            self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
            self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
        if audio_embedding != None:
            if args.trans == 1:
                self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
            else:
                self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        if audio_embedding != None:
            self.audio_embedding = audio_embedding
        else:
            self.audio_embedding = None

        num_modals = 3 if audio_embedding is not None else 2
        gate_hidden = getattr(args, 'gate_dim', 32)
        self.modal_gating = ModalGatingNetwork(args.latdim, num_modals, hidden_dim=gate_hidden)

        self.dropout = nn.Dropout(p=0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)
                
    def getItemEmbeds(self):
        return self.iEmbeds
    
    def getUserEmbeds(self):
        return self.uEmbeds
    
    def getImageFeats(self):
        if args.trans == 0 or args.trans == 2:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            return image_feats
        else:
            return self.image_trans(self.image_embedding)
    
    def getTextFeats(self):
        if args.trans == 0:
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
            return text_feats
        else:
            return self.text_trans(self.text_embedding)

    def getAudioFeats(self):
        if self.audio_embedding == None:
            return None
        else:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)
        return audio_feats

    def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
        if args.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif args.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        if audio_adj != None:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)

        embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(adj, embedsImage)

        embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
        embedsImage_ = torch.spmm(adj, embedsImage_)
        embedsImage += embedsImage_

        embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
        embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(adj, embedsText)

        embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
        embedsText_ = torch.spmm(adj, embedsText_)
        embedsText += embedsText_

        if audio_adj is not None:
            embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
            embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

            embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
            embedsAudio = torch.spmm(adj, embedsAudio)

            embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
            embedsAudio_ = torch.spmm(adj, embedsAudio_)
            embedsAudio += embedsAudio_

        embedsImage += args.ris_adj_lambda * embedsImageAdj
        embedsText += args.ris_adj_lambda * embedsTextAdj
        if audio_adj is not None:
            embedsAudio += args.ris_adj_lambda * embedsAudioAdj

        # --- Dynamic per-node modal gating ---
        # gate_input: trung bình các modal embeddings, shape (N, latdim)
        if audio_adj is None:
            gate_input = (embedsImage + embedsText) / 2.0
        else:
            gate_input = (embedsImage + embedsText + embedsAudio) / 3.0

        # gate_weights: (N, num_modals), mỗi node có bộ trọng số α riêng
        gate_weights = self.modal_gating(gate_input)
        self._last_gate_weights = gate_weights.detach()  # lưu để debug / log

        if audio_adj is None:
            embedsModal = (gate_weights[:, 0:1] * embedsImage
                          + gate_weights[:, 1:2] * embedsText)
        else:
            embedsModal = (gate_weights[:, 0:1] * embedsImage
                          + gate_weights[:, 1:2] * embedsText
                          + gate_weights[:, 2:3] * embedsAudio)

        embeds = embedsModal
        embedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)

        embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

        return embeds[:args.user], embeds[args.user:]

    def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
        if args.trans == 0:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
        elif args.trans == 1:
            image_feats = self.image_trans(self.image_embedding)
            text_feats = self.text_trans(self.text_embedding)
        else:
            image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
            text_feats = self.text_trans(self.text_embedding)

        if audio_adj != None:
            if args.trans == 0:
                audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
            else:
                audio_feats = self.audio_trans(self.audio_embedding)

        embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
        embedsImage = torch.spmm(image_adj, embedsImage)

        embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
        embedsText = torch.spmm(text_adj, embedsText)

        if audio_adj != None:
            embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
            embedsAudio = torch.spmm(audio_adj, embedsAudio)

        embeds1 = embedsImage
        embedsLst1 = [embeds1]
        for gcn in self.gcnLayers:
            embeds1 = gcn(adj, embedsLst1[-1])
            embedsLst1.append(embeds1)
        embeds1 = sum(embedsLst1)

        embeds2 = embedsText
        embedsLst2 = [embeds2]
        for gcn in self.gcnLayers:
            embeds2 = gcn(adj, embedsLst2[-1])
            embedsLst2.append(embeds2)
        embeds2 = sum(embedsLst2)

        if audio_adj != None:
            embeds3 = embedsAudio
            embedsLst3 = [embeds3]
            for gcn in self.gcnLayers:
                embeds3 = gcn(adj, embedsLst3[-1])
                embedsLst3.append(embeds3)
            embeds3 = sum(embedsLst3)

        if audio_adj == None:
            return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
        else:
            return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

    def reg_loss(self):
        ret = 0
        ret += self.uEmbeds.norm(2).square()
        ret += self.iEmbeds.norm(2).square()
        return ret

    def gate_entropy_loss(self, gate_weights):
        """
        Entropy regularization cho gating weights.
        Maximize entropy H(α) = -Σ α_m * log(α_m) để tránh collapse về 1 modal.
        Trả về negative entropy (minimize → maximize entropy).
        Chỉ có hiệu lực khi args.gate_reg > 0.
        """
        eps = 1e-8
        entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=-1).mean()
        return -entropy  # minimize this = maximize entropy

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate

    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
