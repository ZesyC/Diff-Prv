import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Params import MODEL_NAME, args
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import json
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import Model
from DataHandler import DataHandler
from VelocityModel import VelocityModel
from FlowMatching import GraphFlowMatching
from Utils.Utils import contrastLoss, pairPredict
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
        self.test_item_sets = [
            set(items) if items is not None else set()
            for items in self.handler.tstLoader.dataset.tstLocs
        ]

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

        # Overfitting monitoring
        history = {
            'train_loss': [], 'train_bpr': [], 'train_cl': [],
            'train_recall': [], 'train_ndcg': [], 'train_precision': [],
            'test_recall': [], 'test_ndcg': [], 'test_precision': [],
            'test_epochs': [],
        }
        # Early stopping
        no_improve = 0

        log('Model Initialized')

        for ep in range(0, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))

            history['train_loss'].append(reses['Loss'])
            history['train_bpr'].append(reses['BPR Loss'])
            history['train_cl'].append(reses['CL loss'])

            if tstFlag:
                trnReses = self.trainEvalEpoch()
                log(self.makePrint('TrainEval', ep, trnReses, tstFlag))

                history['train_recall'].append(trnReses['Recall'])
                history['train_ndcg'].append(trnReses['NDCG'])
                history['train_precision'].append(trnReses['Precision'])

                reses = self.testEpoch()

                history['test_epochs'].append(ep)
                history['test_recall'].append(reses['Recall'])
                history['test_ndcg'].append(reses['NDCG'])
                history['test_precision'].append(reses['Precision'])

                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    precisionMax = reses['Precision']
                    bestEpoch = ep
                    no_improve = 0
                else:
                    no_improve += 1

                gap = trnReses['Recall'] - reses['Recall']
                trend = 'NEW BEST' if no_improve == 0 else 'no improve %d/%d' % (no_improve, args.patience)
                log(self.makePrint('Test', ep, reses, tstFlag) + '[%s] [gap=%.4f]' % (trend, gap))
            print()

            if args.patience > 0 and no_improve >= args.patience:
                log('Early stopping at epoch %d: no improvement for %d test epochs' % (ep, args.patience))
                break

        self.printOverfitDiagnostic(history, bestEpoch)
        print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)
        return {
            'best_epoch': bestEpoch,
            'recall': recallMax,
            'ndcg': ndcgMax,
            'precision': precisionMax
        }

    def printOverfitDiagnostic(self, history, bestEpoch):
        test_epochs = history['test_epochs']
        test_recall = history['test_recall']
        test_ndcg = history['test_ndcg']

        log('')
        log('=' * 60)
        log('OVERFITTING DIAGNOSTIC')
        log('=' * 60)

        if len(test_recall) < 2:
            log('Not enough test epochs for diagnostic')
            log('=' * 60)
            return

        best_idx = test_recall.index(max(test_recall))
        best_recall = test_recall[best_idx]
        final_recall = test_recall[-1]
        best_ndcg = test_ndcg[best_idx]
        final_ndcg = test_ndcg[-1]
        total_trained = test_epochs[-1] + 1

        log('Best test epoch: %d / %d trained' % (bestEpoch, total_trained))
        log('Recall : %.4f (best) -> %.4f (final), delta = %+.4f' % (best_recall, final_recall, final_recall - best_recall))
        log('NDCG   : %.4f (best) -> %.4f (final), delta = %+.4f' % (best_ndcg, final_ndcg, final_ndcg - best_ndcg))

        if len(history['train_loss']) >= 2:
            log('Train Loss: %.4f (first) -> %.4f (last)' % (history['train_loss'][0], history['train_loss'][-1]))
            log('BPR Loss  : %.4f (first) -> %.4f (last)' % (history['train_bpr'][0], history['train_bpr'][-1]))

        train_recall = history.get('train_recall', [])
        if train_recall:
            final_gap = train_recall[-1] - final_recall
            best_gap = train_recall[best_idx] - best_recall
            log('Train-Test gap: %.4f (at best) -> %.4f (final)' % (best_gap, final_gap))

        recall_drop_pct = (best_recall - final_recall) / best_recall * 100 if best_recall > 0 else 0
        gap_increasing = False
        if len(train_recall) >= 2:
            early_gap = train_recall[0] - test_recall[0]
            late_gap = train_recall[-1] - test_recall[-1]
            gap_increasing = late_gap > early_gap * 1.5

        if recall_drop_pct > 5 or (recall_drop_pct > 1 and gap_increasing):
            verdict = 'OVERFITTING - Recall dropped %.1f%%' % recall_drop_pct
            if gap_increasing:
                verdict += ', train-test gap widening'
        elif recall_drop_pct > 1:
            verdict = 'MILD OVERFITTING - Recall dropped %.1f%% after peak' % recall_drop_pct
        elif bestEpoch >= total_trained - 1:
            verdict = 'POSSIBLY UNDERTRAINED - best epoch is the last epoch'
        else:
            verdict = 'HEALTHY - metrics stable after peak'
        log('VERDICT: %s' % verdict)

        log('')
        header = '%-8s %-14s %-14s %-14s %-14s %-14s %-14s %-8s' % (
            'Epoch', 'Trn Recall', 'Tst Recall', 'Trn NDCG', 'Tst NDCG', 'Trn Prec', 'Tst Prec', 'Gap')
        log(header)
        log('-' * len(header))
        for i, ep in enumerate(test_epochs):
            trn_r = train_recall[i] if i < len(train_recall) else 0
            trn_n = history.get('train_ndcg', [0]*(i+1))[i] if i < len(history.get('train_ndcg', [])) else 0
            trn_p = history.get('train_precision', [0]*(i+1))[i] if i < len(history.get('train_precision', [])) else 0
            gap = trn_r - test_recall[i]
            marker = ' <-- BEST' if ep == bestEpoch else ''
            log('%-8d %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f %-14.4f %-8.4f%s' % (
                ep, trn_r, test_recall[i], trn_n, test_ndcg[i], trn_p, history['test_precision'][i], gap, marker))
        log('=' * 60)

    def prepareModel(self):
        if args.data == 'tiktok':
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).to(device)
        else:
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        self.flow_matching = GraphFlowMatching(sigma_min=1e-4).to(device)
        out_dims = list(args.dims) + [args.item]
        in_dims = out_dims[::-1]
        cond_dim = 0
        if args.modal_cond == 1:
            cond_dim = args.latdim * (2 if args.behavior_cond == 1 else 1)
        self.velocity_model_image = VelocityModel(in_dims, out_dims, args.d_emb_size, cond_dim=cond_dim, norm=args.norm).to(device)
        self.velocity_opt_image = torch.optim.Adam(self.velocity_model_image.parameters(), lr=args.lr, weight_decay=0)
        self.velocity_model_text = VelocityModel(in_dims, out_dims, args.d_emb_size, cond_dim=cond_dim, norm=args.norm).to(device)
        self.velocity_opt_text = torch.optim.Adam(self.velocity_model_text.parameters(), lr=args.lr, weight_decay=0)
        if args.data == 'tiktok':
            self.velocity_model_audio = VelocityModel(in_dims, out_dims, args.d_emb_size, cond_dim=cond_dim, norm=args.norm).to(device)
            self.velocity_opt_audio = torch.optim.Adam(self.velocity_model_audio.parameters(), lr=args.lr, weight_decay=0)

    def buildBehaviorGuidedCond(self, batch_item, batch_index, modal_feats):
        if modal_feats.size(1) != args.latdim:
            raise ValueError(f'Projected modal feature dim must equal latdim={args.latdim}, got {modal_feats.size(1)}')

        interaction_count = batch_item.sum(dim=1, keepdim=True)
        modal_pref = torch.mm(batch_item, modal_feats) / interaction_count.clamp_min(1.0)
        global_modal_pref = modal_feats.mean(dim=0, keepdim=True).expand_as(modal_pref)
        modal_pref = torch.where(interaction_count > 0, modal_pref, global_modal_pref)
        modal_pref = F.normalize(modal_pref, dim=-1)

        if args.behavior_cond == 0:
            return modal_pref

        user_cf = self.model.getUserEmbeds()[batch_index].detach()
        user_cf = F.normalize(user_cf, dim=-1)
        return torch.cat([user_cf, modal_pref], dim=-1)

    def normalizeAdj(self, mat): 
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        cfm_mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32).tocsr()
        train_mat = self.handler.trnMat.tocsr().astype(np.float32)
        mat = train_mat + cfm_mat.multiply(args.cfm_residual_weight)

        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).to(device)

    def initRebuildStats(self):
        return {
            'candidate_edges': 0,
            'generated_edges': 0,
            'filtered_edges': 0,
            'train_overlap': 0,
            'top_score_sum': 0.0,
            'top_score_sq_sum': 0.0,
            'top_score_count': 0,
            'graph_hit': 0,
            'graph_hit_users': 0,
        }

    def updateGraphHitStats(self, stats, scores, batch_index):
        hit_k = min(20, scores.size(1))
        _, hit_indices = torch.topk(scores, k=hit_k)
        hit_indices = hit_indices.cpu().numpy()
        batch_index = batch_index.cpu().numpy()

        for row, user in enumerate(batch_index):
            test_items = self.test_item_sets[int(user)]
            if not test_items:
                continue
            stats['graph_hit_users'] += 1
            if any(int(item) in test_items for item in hit_indices[row]):
                stats['graph_hit'] += 1

    def updateRebuildStats(self, stats, scores, batch_item, batch_index, top_scores, indices_, valid_edges, candidate_edges=None):
        if candidate_edges is None:
            candidate_edges = valid_edges
        selected_train_edges = batch_item.gather(1, indices_).bool() & valid_edges
        valid_scores = top_scores[valid_edges]
        candidate_count = int(candidate_edges.sum().item())
        generated_count = int(valid_edges.sum().item())

        stats['candidate_edges'] += candidate_count
        stats['generated_edges'] += generated_count
        stats['filtered_edges'] += candidate_count - generated_count
        stats['train_overlap'] += int(selected_train_edges.sum().item())
        stats['top_score_count'] += int(valid_scores.numel())
        if valid_scores.numel() > 0:
            stats['top_score_sum'] += float(valid_scores.sum().item())
            stats['top_score_sq_sum'] += float(valid_scores.square().sum().item())

        self.updateGraphHitStats(stats, scores, batch_index)

    def formatRebuildStats(self, modal, stats):
        candidate_edges = stats.get('candidate_edges', stats['generated_edges'])
        generated_edges = stats['generated_edges']
        kept_ratio = generated_edges / candidate_edges if candidate_edges > 0 else 0.0
        train_overlap_ratio = stats['train_overlap'] / generated_edges if generated_edges > 0 else 0.0
        new_edge_ratio = 1.0 - train_overlap_ratio if generated_edges > 0 else 0.0
        score_count = stats['top_score_count']
        top_score_mean = stats['top_score_sum'] / score_count if score_count > 0 else 0.0
        top_score_var = stats['top_score_sq_sum'] / score_count - top_score_mean ** 2 if score_count > 0 else 0.0
        top_score_std = float(np.sqrt(max(top_score_var, 0.0)))
        graph_hit = stats['graph_hit'] / stats['graph_hit_users'] if stats['graph_hit_users'] > 0 else 0.0

        return (
            'Rebuild %s: rebuild_k = %d, candidate_edges = %d, generated_edges = %d, kept_ratio = %.6f, '
            'train_overlap_ratio = %.6f, new_edge_ratio = %.6f, '
            'top_score_mean = %.6f, top_score_std = %.6f, graph_hit@20 = %.6f'
            % (
                modal,
                args.rebuild_k,
                candidate_edges,
                generated_edges,
                kept_ratio,
                train_overlap_ratio,
                new_edge_ratio,
                top_score_mean,
                top_score_std,
                graph_hit,
            )
        )

    def collectRebuildEdges(self, scores, batch_item, batch_index, stats=None):
        scores = scores.masked_fill(batch_item.bool(), float('-inf'))
        top_k = min(args.rebuild_k, scores.size(1))
        top_scores, indices_ = torch.topk(scores, k=top_k)

        positive_scores = top_scores.clamp_min(0.0)
        edge_weights = positive_scores / (positive_scores + 1.0)
        candidate_edges = torch.isfinite(top_scores) & (edge_weights > 0)
        valid_edges = candidate_edges
        if args.rebuild_min_score is not None:
            valid_edges = valid_edges & (top_scores >= args.rebuild_min_score)
        batch_users = batch_index.unsqueeze(1).expand_as(indices_)

        if stats is not None:
            self.updateRebuildStats(stats, scores, batch_item, batch_index, top_scores, indices_, valid_edges, candidate_edges)

        return (
            batch_users[valid_edges].cpu().numpy(),
            indices_[valid_edges].cpu().numpy(),
            edge_weights[valid_edges].cpu().numpy(),
        )

    def trainEpoch(self):
        self.model.train()
        self.velocity_model_image.train()
        self.velocity_model_text.train()
        if args.data == 'tiktok':
            self.velocity_model_audio.train()

        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, epClLoss = 0, 0, 0
        epDiLoss_image, epDiLoss_text = 0, 0
        if args.data == 'tiktok':
            epDiLoss_audio = 0
        steps = len(trnLoader)

        cfmLoader = self.handler.cfmLoader
        cfmEvalLoader = self.handler.cfmEvalLoader
        cfm_steps = len(cfmLoader)

        for i, batch in enumerate(cfmLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.to(device), batch_index.to(device)

            iEmbeds = self.model.getItemEmbeds().detach()

            image_feats = self.model.getImageFeats().detach()
            text_feats = self.model.getTextFeats().detach()
            if args.data == 'tiktok':
                audio_feats = self.model.getAudioFeats().detach()

            self.velocity_opt_image.zero_grad()
            self.velocity_opt_text.zero_grad()
            if args.data == 'tiktok':
                self.velocity_opt_audio.zero_grad()

            if args.modal_cond == 1:
                image_cond = self.buildBehaviorGuidedCond(batch_item, batch_index, image_feats).detach()
                text_cond = self.buildBehaviorGuidedCond(batch_item, batch_index, text_feats).detach()
                if args.data == 'tiktok':
                    audio_cond = self.buildBehaviorGuidedCond(batch_item, batch_index, audio_feats).detach()
            else:
                image_cond = text_cond = None
                if args.data == 'tiktok':
                    audio_cond = None

            cfm_loss_image, msi_loss_image, alpha_hat_image = self.flow_matching.training_losses(
                self.velocity_model_image,
                batch_item,
                iEmbeds,
                batch_index,
                image_feats,
                modal_cond=image_cond,
                cfm_lambda=args.cfm_lambda,
            )
            cfm_loss_text, msi_loss_text, alpha_hat_text = self.flow_matching.training_losses(
                self.velocity_model_text,
                batch_item,
                iEmbeds,
                batch_index,
                text_feats,
                modal_cond=text_cond,
                cfm_lambda=args.cfm_lambda,
            )
            if args.data == 'tiktok':
                cfm_loss_audio, msi_loss_audio, alpha_hat_audio = self.flow_matching.training_losses(
                    self.velocity_model_audio,
                    batch_item,
                    iEmbeds,
                    batch_index,
                    audio_feats,
                    modal_cond=audio_cond,
                    cfm_lambda=args.cfm_lambda,
                )

            loss_image = cfm_loss_image.mean() + msi_loss_image.mean() * args.e_loss
            loss_text = cfm_loss_text.mean() + msi_loss_text.mean() * args.e_loss
            if args.data == 'tiktok':
                loss_audio = cfm_loss_audio.mean() + msi_loss_audio.mean() * args.e_loss

            cross_fm_loss = torch.tensor(0.0, device=device)
            if args.cross_fm_weight > 0:
                sim_it = F.cosine_similarity(alpha_hat_image.unsqueeze(1), alpha_hat_text.unsqueeze(0), dim=-1) / args.temp
                labels = torch.arange(batch_index.shape[0], device=device)
                cross_fm_loss = cross_fm_loss + F.cross_entropy(sim_it, labels)
                if args.data == 'tiktok':
                    sim_ia = F.cosine_similarity(alpha_hat_image.unsqueeze(1), alpha_hat_audio.unsqueeze(0), dim=-1) / args.temp
                    sim_ta = F.cosine_similarity(alpha_hat_text.unsqueeze(1), alpha_hat_audio.unsqueeze(0), dim=-1) / args.temp
                    cross_fm_loss = cross_fm_loss + F.cross_entropy(sim_ia, labels) + F.cross_entropy(sim_ta, labels)
                cross_fm_loss = cross_fm_loss * args.cross_fm_weight

            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            if args.data == 'tiktok':
                epDiLoss_audio += loss_audio.item()

            if args.data == 'tiktok':
                loss = loss_image + loss_text + loss_audio
            else:
                loss = loss_image + loss_text

            loss = loss + cross_fm_loss

            loss.backward()

            self.velocity_opt_image.step()
            self.velocity_opt_text.step()
            if args.data == 'tiktok':
                self.velocity_opt_audio.step()
            log('CFM Step %d/%d' % (i + 1, cfm_steps), save=False, oneline=True)

        log('')
        log('Start to re-build UI matrix using Euler Solver')

        self.velocity_model_image.eval()
        self.velocity_model_text.eval()
        if args.data == 'tiktok':
            self.velocity_model_audio.eval()

        rebuild_generator = torch.Generator(device=device)
        rebuild_generator.manual_seed(args.seed)

        with torch.no_grad():
            image_feats = self.model.getImageFeats().detach()
            text_feats = self.model.getTextFeats().detach()
            if args.data == 'tiktok':
                audio_feats = self.model.getAudioFeats().detach()

            u_list_image = []
            i_list_image = []
            edge_list_image = []
            rebuild_stats_image = self.initRebuildStats()

            u_list_text = []
            i_list_text = []
            edge_list_text = []
            rebuild_stats_text = self.initRebuildStats()

            if args.data == 'tiktok':
                u_list_audio = []
                i_list_audio = []
                edge_list_audio = []
                rebuild_stats_audio = self.initRebuildStats()

            for _, batch in enumerate(cfmEvalLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.to(device), batch_index.to(device)

                x_start = torch.randn(
                    batch_item.shape,
                    dtype=batch_item.dtype,
                    device=device,
                    generator=rebuild_generator,
                )

                if args.modal_cond == 1:
                    image_cond_inf = self.buildBehaviorGuidedCond(batch_item, batch_index, image_feats).detach()
                    text_cond_inf = self.buildBehaviorGuidedCond(batch_item, batch_index, text_feats).detach()
                    if args.data == 'tiktok':
                        audio_cond_inf = self.buildBehaviorGuidedCond(batch_item, batch_index, audio_feats).detach()
                else:
                    image_cond_inf = text_cond_inf = None
                    if args.data == 'tiktok':
                        audio_cond_inf = None

                denoised_batch_image = self.flow_matching.euler_solve(self.velocity_model_image, x_start, steps=args.steps, cond=image_cond_inf)
                denoised_batch_text = self.flow_matching.euler_solve(self.velocity_model_text, x_start, steps=args.steps, cond=text_cond_inf)
                if args.data == 'tiktok':
                    denoised_batch_audio = self.flow_matching.euler_solve(self.velocity_model_audio, x_start, steps=args.steps, cond=audio_cond_inf)

                batch_users, batch_items, edge_weights = self.collectRebuildEdges(
                    denoised_batch_image, batch_item, batch_index, rebuild_stats_image
                )
                u_list_image.append(batch_users)
                i_list_image.append(batch_items)
                edge_list_image.append(edge_weights)

                batch_users, batch_items, edge_weights = self.collectRebuildEdges(
                    denoised_batch_text, batch_item, batch_index, rebuild_stats_text
                )
                u_list_text.append(batch_users)
                i_list_text.append(batch_items)
                edge_list_text.append(edge_weights)

                if args.data == 'tiktok':
                    batch_users, batch_items, edge_weights = self.collectRebuildEdges(
                        denoised_batch_audio, batch_item, batch_index, rebuild_stats_audio
                    )
                    u_list_audio.append(batch_users)
                    i_list_audio.append(batch_items)
                    edge_list_audio.append(edge_weights)

            log(self.formatRebuildStats('image', rebuild_stats_image))
            log(self.formatRebuildStats('text', rebuild_stats_text))
            if args.data == 'tiktok':
                log(self.formatRebuildStats('audio', rebuild_stats_audio))

            u_list_image = np.concatenate(u_list_image)
            i_list_image = np.concatenate(i_list_image)
            edge_list_image = np.concatenate(edge_list_image)
            self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

            u_list_text = np.concatenate(u_list_text)
            i_list_text = np.concatenate(i_list_text)
            edge_list_text = np.concatenate(edge_list_text)
            self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

            if args.data == 'tiktok':
                u_list_audio = np.concatenate(u_list_audio)
                i_list_audio = np.concatenate(i_list_audio)
                edge_list_audio = np.concatenate(edge_list_audio)
                self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
                self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

        log('UI matrix built!')

        self.velocity_model_image.train()
        self.velocity_model_text.train()
        if args.data == 'tiktok':
            self.velocity_model_audio.train()

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
            bprLoss = - (scoreDiff).sigmoid().log().sum() / ancs.shape[0]
            regLoss = self.model.reg_loss() * args.reg
            loss = bprLoss + regLoss

            if args.gate_reg > 0 and hasattr(self.model, '_last_gate_weights'):
                gateLoss = self.model.gate_entropy_loss(self.model._last_gate_weights) * args.gate_reg
                loss = loss + gateLoss
            else:
                gateLoss = torch.tensor(0.0, device=device)

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

            log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ; gate : %.3f' % (
                i + 1,
                steps,
                bprLoss.item(),
                regLoss.item(),
                clLoss.item(),
                gateLoss.item() if hasattr(gateLoss, 'item') else gateLoss
                ), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPR Loss'] = epRecLoss / steps
        ret['CL loss'] = epClLoss / steps
        ret['CFM image loss'] = epDiLoss_image / cfm_steps
        ret['CFM text loss'] = epDiLoss_text / cfm_steps
        if args.data == 'tiktok':
            ret['CFM audio loss'] = epDiLoss_audio / cfm_steps
        return ret

    def trainEvalEpoch(self, sample_size=1024):
        self.model.eval()
        with torch.no_grad():
            if args.data == 'tiktok':
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

            trnMat = self.handler.trnMat.tocsr()
            active_users = np.array([u for u in range(args.user) if trnMat[u].nnz > 0])
            sample_size = min(sample_size, len(active_users))
            sampled = np.random.choice(active_users, sample_size, replace=False)

            trnLocs = [None] * args.user
            for u in sampled:
                trnLocs[u] = list(trnMat[u].indices)

            epRecall = epNdcg = epPrecision = 0
            batch_size = args.tstBat
            for start in range(0, len(sampled), batch_size):
                batch_users = sampled[start:start + batch_size]
                usr = torch.LongTensor(batch_users).to(device)
                allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0))
                _, topLocs = torch.topk(allPreds, args.topk)
                recall, ndcg, precision = self.calcRes(
                    topLocs.cpu().numpy(), trnLocs, batch_users)
                epRecall += recall
                epNdcg += ndcg
                epPrecision += precision

            ret = dict()
            ret['Recall'] = epRecall / sample_size
            ret['NDCG'] = epNdcg / sample_size
            ret['Precision'] = epPrecision / sample_size
        self.model.train()
        return ret

    def testEpoch(self):
        self.model.eval()

        with torch.no_grad():
            tstLoader = self.handler.tstLoader
            epRecall, epNdcg, epPrecision = [0] * 3
            i = 0
            num = tstLoader.dataset.__len__()
            steps = len(tstLoader)

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
                recall, ndcg, precision = self.calcRes(
                    topLocs.cpu().numpy(),
                    self.handler.tstLoader.dataset.tstLocs,
                    usr.cpu().numpy(),
                )
                epRecall += recall
                epNdcg += ndcg
                epPrecision += precision
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)

            ret = dict()
            ret['Recall'] = epRecall / num
            ret['NDCG'] = epNdcg / num
            ret['Precision'] = epPrecision / num

        self.model.train()
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = allPrecision = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[int(batIds[i])]
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

if __name__ == '__main__':
    seed_it(args.seed)

    logger.saveDefault = True
    
    log(f'Start {MODEL_NAME}')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    results = coach.run()
    
    out_file = f"results_{MODEL_NAME}_{args.data}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f)
    print(f"Results saved to {out_file}")
