import sys
import pickle as pkl
import gzip as gz
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import glob


import drawTemplate as dt

def readEdge(graph, layer):
	with gz.open(f'{graph}_{layer}_intersection.w-wcc--l-l.pkl.gz', 'rb') as f:
		edgeDict = pkl.load(f)
	return edgeDict

def getLayout(edgeSet):
	G = nx.Graph()
	G.add_edges_from(edgeSet)
	isPlanar, _ = nx.check_planarity(G)
	if isPlanar:
		pos = nx.planar_layout(G)
	else:
		pos = nx.spring_layout(G)

	# print(pos)
	# nx.draw(G, pos)
	# plt.show()
	return pos, isPlanar

class drawing(object):
	"""docstring for drawing"""
	def __init__(self, pos, edgeSet, scale = 250, nodeR = 50):
		super(drawing, self).__init__()
		self.pos = pos
		self.edgeSet = edgeSet
		self.scale = scale - nodeR
		self.nodeR = nodeR
		self.rotation = np.random.uniform(np.pi * 2)

		c, s = np.cos(self.rotation), np.sin(self.rotation)
		self.rMatrix = np.array([[c, -s], [s, c]])
		self.rotatePos()
		return

	def rotatePos(self):
		for node, xy in self.pos.items():
			xy = np.matmul(self.rMatrix, xy)
			xy *= self.scale
			xy += self.scale
			self.pos[node] = xy

		XList = [x for x, y in self.pos.values()]
		YList = [y for x, y in self.pos.values()]
		minX = min(XList)
		minY = min(YList)
		maxX = max(XList)
		maxY = max(YList)
		XYRatio = self.scale * 2 / np.array((maxX - minX, maxY - minY), dtype = np.float32)
		for node, xy in self.pos.items():
			self.pos[node] = xy * XYRatio
		return

	def drawNode(self):
		strokeList = []
		self.minX = 0
		self.minY = 0
		self.ratioDict = dict()
		for node, xy in self.pos.items():
			label = f'F{node}'
			if node > 2:
				if np.random.uniform() < 0.5:
					label = f'K{node + 1}'

			offSet = []
			labelPosListList = []
			tempX = 0
			tempY = 0
			for char in label:
				# print(f'****{char = }')
				charTemplate = dt.label(char)
				posList, region = charTemplate.getPosList()
				labelPosListList.append(posList)
				offSet.append(np.array([tempX, 0], dtype = np.int32))
				tempX += region[1][0]
				tempY = max(tempY, region[1][1])
				# for stroke in posList:
				# 	strokeList.append((stroke[:, 0], stroke[:, 1], np.full(stroke.shape[0], 2, dtype = np.int32)))
			
			nodeTemplate = dt.node()
			posList, region = nodeTemplate.getPosList()
			region = np.array(region, dtype = np.int32)
			regionSize = region[1] - region[0]
			ratio = np.max(np.array([tempX, tempY], dtype = np.float32) / regionSize)
			ratio = max(ratio, 1)
			# print(ratio)
			for idx, stroke in enumerate(posList):
				posList[idx] = np.round(stroke * ratio).astype(np.int32)
				self.minX = min(self.minX, np.min(posList[idx][:, 0]))
				self.minY = min(self.minY, np.min(posList[idx][:, 1]))
			region = np.round(region * ratio).astype(np.int32) + xy
			self.ratioDict[node] = ratio

			for stroke in posList:
				strokeList.append((stroke[:, 0] + xy[0], stroke[:, 1] + xy[1], np.full(stroke.shape[0], 0, dtype = np.int32)))
			for labelIdx, labelPosList in enumerate(labelPosListList):
				for stroke in labelPosList:
					strokeList.append((stroke[:, 0] + region[0, 0] + offSet[labelIdx][0], stroke[:, 1] + region[0, 1] + offSet[labelIdx][1], np.full(stroke.shape[0], 2, dtype = np.int32)))
		return strokeList

	def adjustMin(self, strokeList):
		if self.minX < 0:
			print('adjust X')
			for strokeIdx, stroke in enumerate(strokeList):
				strokeList[strokeIdx][0] -= self.minX
		if self.minY < 0:
			print('adjust Y')
			for strokeIdx, stroke in enumerate(strokeList):
				strokeList[strokeIdx][1] -= self.minY
		return strokeList

	def drawLink(self):
		strokeList = []
		for (src, tgt) in self.edgeSet:
			srcPos = np.round(self.pos[src] + self.nodeR * self.ratioDict[src]).astype(np.int32)
			tgtPos = np.round(self.pos[tgt] + self.nodeR * self.ratioDict[tgt]).astype(np.int32)
			vec = tgtPos - srcPos
			dist = np.sqrt(np.sum(np.square(vec)))
			unit = vec / dist
			# print(self.ratioDict[src] * unit)
			start = np.round(srcPos + self.ratioDict[src] * self.nodeR * unit).astype(np.int32)
			end = (tgtPos - self.ratioDict[tgt] * self.nodeR * unit).astype(np.int32)
			linkTemplate = dt.link(start, end)
			posList = linkTemplate.getPosList()
			for stroke in posList:
				strokeList.append((stroke[:, 0], stroke[:, 1], np.full(stroke.shape[0], 1, dtype = np.int32)))
		return strokeList


def showDrawing(strokeList):
	# define the colormap
	cmap = plt.cm.jet
	# extract all colors from the .jet map
	cmaplist = [cmap(i) for i in range(3)]
	# create the new map
	cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
	for stroke in strokeList:
		X = stroke[0]
		Y = stroke[1]
		label = stroke[2][0]

		plt.plot(X, -Y, c = f'C{label}')
	plt.xlim(0, 800)
	plt.ylim(-800, 0)

if __name__ == '__main__':
	graph = sys.argv[1]

	planarDict = dict()
	nonPlanarDict = dict()

	layerFileList = glob.glob(f'{graph}_*_intersection.w-wcc--l-l.pkl.gz')

	layerList = [int(layerFile.split('_')[1]) for layerFile in layerFileList]

	for layer in layerList:
		edgeDict = readEdge(graph, layer)

		for (wave, wcc), edgeSet in edgeDict.items():
			print(f'**{layer = }, {wave = }, {wcc = }')
			pos, isPlanar = getLayout(edgeSet)
			sketch = drawing(pos, edgeSet)
			strokeList = sketch.drawNode()
			strokeList.extend(sketch.drawLink())
			strokeList = sketch.adjustMin(strokeList)
			showDrawing(strokeList)
			if isPlanar:
				plt.savefig(f'planar/{graph}_{layer}_{wave}_{wcc}_intersection.png')
				planarDict[(graph, layer, wave, wcc)] = strokeList
			else:
				plt.savefig(f'non-planar/{graph}_{layer}_{wave}_{wcc}_intersection.png')
				nonPlanarDict[(graph, layer, wave, wcc)] = strokeList
			plt.clf()

		with gz.open(f'planar/{graph}_intersection_sketch.g-l-w-wcc--x-y-l.pkl.gz', 'wb') as f:
			pkl.dump(planarDict, f)
		with gz.open(f'non-planar/{graph}_intersection_sketch.g-l-w-wcc--x-y-l.pkl.gz', 'wb') as f:
			pkl.dump(nonPlanarDict, f)