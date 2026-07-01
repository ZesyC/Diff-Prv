import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
import os
import zipfile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
class DataHandler:
	def __init__(self):
		if args.data not in {'baby', 'tiktok', 'sports'}:
			raise ValueError(f"Unsupported dataset '{args.data}'. Supported datasets: baby, tiktok, sports")
		predir = os.path.join(_project_root, 'Datasets', args.data) + '/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'
		if args.data == 'tiktok':
			self.audiofile = predir + 'audio_feat.npy'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
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

	def loadFeatures(self, filename):
		if os.path.exists(filename):
			feats = np.load(filename)
		elif os.path.exists(filename + '.zip'):
			with zipfile.ZipFile(filename + '.zip') as zf:
				with zf.open(os.path.basename(filename)) as fs:
					feats = np.load(fs)
		else:
			raise FileNotFoundError(f'Feature file not found: {filename}')
		return torch.tensor(feats).float().to(device), np.shape(feats)[1]

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
		if args.data == 'tiktok':
			self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)

		self.cfmData = CFMData(torch.FloatTensor(self.trnMat.toarray()))
		self.cfmLoader = dataloader.DataLoader(self.cfmData, batch_size=args.batch, shuffle=True, num_workers=0)
		self.cfmEvalLoader = dataloader.DataLoader(self.cfmData, batch_size=args.batch, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		self.negs = np.random.randint(0, args.item, len(self.rows)).astype(np.int32)
		for i in range(len(self.rows)):
			u = self.rows[i]
			while (u, self.negs[i]) in self.dokmat:
				self.negs[i] = np.random.randint(args.item)

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
	
class CFMData(data.Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)
