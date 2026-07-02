import torch
from torch import nn
import torch.nn.functional as F
from DiffMM.Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class CrossModalAttention(nn.Module):
	def __init__(self, latdim, num_heads):
		super(CrossModalAttention, self).__init__()
		if latdim % num_heads != 0:
			raise ValueError('cross_heads must divide latdim')
		self.attn = nn.MultiheadAttention(latdim, num_heads, batch_first=True)
		self.norm = nn.LayerNorm(latdim)

	def forward(self, feats):
		modal_tokens = torch.stack(feats, dim=1)
		attn_tokens, _ = self.attn(modal_tokens, modal_tokens, modal_tokens, need_weights=False)
		attn_tokens = self.norm(modal_tokens + attn_tokens)
		return [attn_tokens[:, i, :] for i in range(attn_tokens.shape[1])]

class ModalityInteraction(nn.Module):
	def __init__(self, latdim):
		super(ModalityInteraction, self).__init__()
		self.gate = nn.Linear(latdim * 2, latdim)
		self.norm = nn.LayerNorm(latdim)

	def forward(self, feats):
		if len(feats) < 2:
			return feats

		stacked_feats = torch.stack(feats, dim=1)
		sum_feats = stacked_feats.sum(dim=1)
		interacted_feats = []
		for i, feat in enumerate(feats):
			context = (sum_feats - feat) / (len(feats) - 1)
			gate = torch.sigmoid(self.gate(torch.cat([feat, context], dim=-1)))
			interacted_feats.append(self.norm(feat + gate * context))
		return interacted_feats

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

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.cross_attention = CrossModalAttention(args.latdim, args.cross_heads) if args.use_cross_attention else None
		self.modality_interaction = ModalityInteraction(args.latdim) if args.use_modality_interaction else None
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds

	def _project_modal_feats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
		else:
			image_feats = self.image_trans(self.image_embedding)

		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		else:
			text_feats = self.text_trans(self.text_embedding)

		feats = [image_feats, text_feats]
		if self.audio_embedding != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
			feats.append(audio_feats)

		return feats

	def _get_modal_feats(self):
		feats = self._project_modal_feats()
		if self.cross_attention is not None:
			attn_feats = self.cross_attention(feats)
			feats = [feat + args.cross_lambda * (attn_feat - feat) for feat, attn_feat in zip(feats, attn_feats)]
		if self.modality_interaction is not None:
			interacted_feats = self.modality_interaction(feats)
			feats = [feat + args.cross_lambda * (interacted_feat - feat) for feat, interacted_feat in zip(feats, interacted_feats)]
		return feats

	def getModalFeats(self):
		return self._get_modal_feats()
	
	def getImageFeats(self):
		return self._get_modal_feats()[0]
	
	def getTextFeats(self):
		return self._get_modal_feats()[1]

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		return self._get_modal_feats()[2]

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		modal_feats = self._get_modal_feats()
		image_feats = modal_feats[0]
		text_feats = modal_feats[1]
		if audio_adj != None:
			audio_feats = modal_feats[2]

		weight = self.softmax(self.modal_weight)

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

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		modal_feats = self._get_modal_feats()
		image_feats = modal_feats[0]
		text_feats = modal_feats[1]
		if audio_adj != None:
			audio_feats = modal_feats[2]

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
		
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, ts)

		mse = self.mean_flat((x_start - model_output) ** 2)

		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
