import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
from VelocityModel import VelocityModel
from FlowMatching import GraphFlowMatching
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
from scipy.sparse import coo_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recallMax = 0
        ndcgMax = 0
        precisionMax = 0
        bestEpoch = 0

        log('Model Initialized')

        for ep in range(0, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    precisionMax = reses['Precision']
                    bestEpoch = ep
                log(self.makePrint('Test', ep, reses, tstFlag))
            print()
        print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)
        return {
            'best_epoch': bestEpoch,
            'recall': recallMax,
            'ndcg': ndcgMax,
            'precision': precisionMax
        }

    def prepareModel(self):
        if args.data == 'tiktok':
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).to(device)
        else:
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        self.flow_matching = GraphFlowMatching(sigma_min=1e-4).to(device)
        out_dims = eval(args.dims) + [args.item]
        in_dims = out_dims[::-1]
        cond_dim = args.latdim * 3
        self.velocity_model_image = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm, cond_dim=cond_dim).to(device)
        self.velocity_opt_image = torch.optim.Adam(self.velocity_model_image.parameters(), lr=args.lr, weight_decay=0)
        self.velocity_model_text = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm, cond_dim=cond_dim).to(device)
        self.velocity_opt_text = torch.optim.Adam(self.velocity_model_text.parameters(), lr=args.lr, weight_decay=0)
        if args.data == 'tiktok':
            self.velocity_model_audio = VelocityModel(in_dims, out_dims, args.d_emb_size, norm=args.norm, cond_dim=cond_dim).to(device)
            self.velocity_opt_audio = torch.optim.Adam(self.velocity_model_audio.parameters(), lr=args.lr, weight_decay=0)

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, epClLoss = 0, 0, 0
        epDiLoss_image, epDiLoss_text = 0, 0
        if args.data == 'tiktok':
            epDiLoss_audio = 0
        steps = trnLoader.dataset.__len__() // args.batch

        cfmLoader = self.handler.cfmLoader

        for i, batch in enumerate(cfmLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(device), batch_index.to(device)

            iEmbeds = self.model.getItemEmbeds().detach()
            uEmbeds = self.model.getUserEmbeds().detach()

            image_feats = self.model.getImageFeats().detach()
            text_feats = self.model.getTextFeats().detach()
            if args.data == 'tiktok':
                audio_feats = self.model.getAudioFeats().detach()

            self.velocity_opt_image.zero_grad()
            self.velocity_opt_text.zero_grad()
            if args.data == 'tiktok':
                self.velocity_opt_audio.zero_grad()

            cfm_loss_image, msi_loss_image = self.flow_matching.training_losses(self.velocity_model_image, batch_item, iEmbeds, uEmbeds, batch_index, image_feats)
            cfm_loss_text, msi_loss_text = self.flow_matching.training_losses(self.velocity_model_text, batch_item, iEmbeds, uEmbeds, batch_index, text_feats)
            if args.data == 'tiktok':
                cfm_loss_audio, msi_loss_audio = self.flow_matching.training_losses(self.velocity_model_audio, batch_item, iEmbeds, uEmbeds, batch_index, audio_feats)

            loss_image = cfm_loss_image.mean() + msi_loss_image.mean() * args.e_loss
            loss_text = cfm_loss_text.mean() + msi_loss_text.mean() * args.e_loss
            if args.data == 'tiktok':
                loss_audio = cfm_loss_audio.mean() + msi_loss_audio.mean() * args.e_loss

            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            if args.data == 'tiktok':
                epDiLoss_audio += loss_audio.item()

            if args.data == 'tiktok':
                loss = loss_image + loss_text + loss_audio
            else:
                loss = loss_image + loss_text

            loss.backward()

            self.velocity_opt_image.step()
            self.velocity_opt_text.step()
            if args.data == 'tiktok':
                self.velocity_opt_audio.step()
            log('CFM Step %d/%d' % (i, cfmLoader.dataset.__len__() // args.batch), save=False, oneline=True)

        log('')
        log('Start to re-build UI matrix using Euler Solver')

        with torch.no_grad():

            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []

            if args.data == 'tiktok':
                u_list_audio = []
                i_list_audio = []
                edge_list_audio = []

            for _, batch in enumerate(cfmLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(device), batch_index.to(device)

                uEmbeds = self.model.getUserEmbeds().detach()
                image_feats = self.model.getImageFeats().detach()
                text_feats = self.model.getTextFeats().detach()
                if args.data == 'tiktok':
                    audio_feats = self.model.getAudioFeats().detach()

                x_start = torch.randn_like(batch_item)
                denoised_batch_image = self.flow_matching.euler_solve(self.velocity_model_image, x_start, batch_item, uEmbeds, batch_index, image_feats, steps=args.steps)
                denoised_batch_text = self.flow_matching.euler_solve(self.velocity_model_text, x_start, batch_item, uEmbeds, batch_index, text_feats, steps=args.steps)
                if args.data == 'tiktok':
                    denoised_batch_audio = self.flow_matching.euler_solve(self.velocity_model_audio, x_start, batch_item, uEmbeds, batch_index, audio_feats, steps=args.steps)

                top_item, indices_ = torch.topk(denoised_batch_image, k=args.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                top_item, indices_ = torch.topk(denoised_batch_text, k=args.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]): 
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)

                if args.data == 'tiktok':
                    top_item, indices_ = torch.topk(denoised_batch_audio, k=args.rebuild_k)
                    for i in range(batch_index.shape[0]):
                        for j in range(indices_[i].shape[0]): 
                            u_list_audio.append(int(batch_index[i].cpu().numpy()))
                            i_list_audio.append(int(indices_[i][j].cpu().numpy()))
                            edge_list_audio.append(1.0)

            # image
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

            # text
            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

            if args.data == 'tiktok':
                # audio
                u_list_audio = np.array(u_list_audio)
                i_list_audio = np.array(i_list_audio)
                edge_list_audio = np.array(edge_list_audio)
                self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
                self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

        log('UI matrix built!')

        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().to(device)
            poss = poss.long().to(device)
            negs = negs.long().to(device)

            self.opt.zero_grad()

            if args.data == 'tiktok':
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = self.model.reg_loss() * args.reg
            loss = bprLoss + regLoss
            
            epRecLoss += bprLoss.item()
            epLoss += loss.item()

            if args.data == 'tiktok':
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
            if args.data == 'tiktok':
                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
            else:
                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

            clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
            clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
            if args.data == 'tiktok':
                clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
                clLoss_ = clLoss1 + clLoss2 + clLoss3
            else:
                clLoss_ = clLoss1 + clLoss2

            if args.cl_method == 1:
                clLoss = clLoss_

            loss += clLoss

            epClLoss += clLoss.item()

            loss.backward()
            self.opt.step()

            log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
                i, 
                steps,
                bprLoss.item(),
                regLoss.item(),
                clLoss.item()
                ), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPR Loss'] = epRecLoss / steps
        ret['CL loss'] = epClLoss / steps
        ret['CFM image loss'] = epDiLoss_image / (cfmLoader.dataset.__len__() // args.batch)
        ret['CFM text loss'] = epDiLoss_text / (cfmLoader.dataset.__len__() // args.batch)
        if args.data == 'tiktok':
            ret['CFM audio loss'] = epDiLoss_audio / (cfmLoader.dataset.__len__() // args.batch)
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        if args.data == 'tiktok':
            usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
        else:
            usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().to(device)
            trnMask = trnMask.to(device)
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, args.topk)
            recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            epPrecision += precision
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        ret['Precision'] = epPrecision / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = allPrecision = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = precision = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
                    precision += 1
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            precision = precision / args.topk
            allRecall += recall
            allNdcg += ndcg
            allPrecision += precision
        return allRecall, allNdcg, allPrecision

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

if __name__ == '__main__':
    seed_it(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    results = coach.run()
    
    import json
    out_file = f"results_cfm_optimized_{args.data}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {out_file}")
