import sys
import json
import glob
import os
from collections import defaultdict
import networkx as nx
import pandas as pd
from itertools import combinations
import pickle as pkl
import gzip as gz
import numpy as np

def getWccListDict(path, graph, eSizeTh):
	def getLayer(waveInfoFile):
		fileName = os.path.split(waveInfoFile)[1]
		layer = int(fileName.split('-')[1])
		return layer

	wccDict = defaultdict(list)
	waveInfoList = sorted(glob.glob(f'{path}{graph}/{graph}_waves/layer-*-waves-info.json'), key = lambda x: getLayer(x), reverse = True)
	
	for waveInfoFile in waveInfoList:
		layer = getLayer(waveInfoFile)
		if layer == 1:
			continue #trees
		
		print(f'--{layer = }')

		with open(waveInfoFile) as f:
			info = json.load(f)
			for wave, waveInfo in info.items():
				wave = int(wave)
				if wave == 0:
					continue

				del waveInfo['vertices']
				del waveInfo['edges']

				for wcc, wccInfo in waveInfo.items():
					wcc = int(wcc)
					if wccInfo['edges'] < 64:
						continue
					if wccInfo['edges'] > eSizeTh:
						continue # large
					if wccInfo['edges'] == wccInfo['vertices'] * (wccInfo['vertices'] - 1) / 2:
						continue # clique
					if wccInfo['edges'] == wccInfo['vertices'] - 1:
						continue # tree

					wccDict[layer].append((wave, wcc))
	return wccDict


def getGraph(layer, wccList, path, graph):
	intersectionEdgeDict = dict()
	df = pd.read_csv(f'{path}{graph}/{graph}_waves/layer-{layer}-waves.csv', header = None, names = ['src', 'tgt', 'wave', 'wcc', 'frag'], usecols = ['src', 'tgt', 'wave', 'wcc'], dtype = np.uint32)
	df = df.set_index(['wave', 'wcc'])
	for w, wcc in wccList:
		print(f'**{layer = }, {w = }, {wcc = }', end = ', ')
		edgeList = df.loc[[(w, wcc)]]
		G = nx.Graph()
		G.add_edges_from(edgeList.to_numpy())
		v2lccDict, fpCount = getFp(G)
		print(f'{fpCount = }')
		if fpCount <= 2:
			continue
		edgeSet = getIntersection(v2lccDict)
		intersectionEdgeDict[(w, wcc)] = edgeSet
	return intersectionEdgeDict

def getFp(G):
	v2lccDict = defaultdict(list)
	fpCount = 0
	while G.size():
		coreDict = nx.core_number(G)
		maxCore = max(coreDict.values())
		vSet = set()
		for v, core in coreDict.items():
			if core == maxCore:
				vSet.add(v)
		coreGraph = G.subgraph(vSet)
		for lccIdx, vSet in enumerate(nx.connected_components(G)):
			for v in vSet:
				v2lccDict[v].append((maxCore, lccIdx))
		G.remove_edges_from(coreGraph.edges())
		fpCount += 1

	return v2lccDict, fpCount

def getIntersection(v2layerDict):
	edgeSet = set()
	for v, lccList in v2layerDict.items():
		for (layerX, lccX), (layerY, lccY) in combinations(lccList, 2):
			if layerX > layerY:
				# edgeSet.add((layerX, lccX, layerY, lccY))
				edgeSet.add((layerX, layerY))
	return edgeSet

def dumpEdges(layer, intersectionEdgeDict, graph):
	with gz.open(f'{graph}_{layer}_intersection.w-wcc--l-l.pkl.gz', 'wb') as f:
		pkl.dump(intersectionEdgeDict, f)
	return

if __name__ == '__main__':
	path = sys.argv[1]
	graph = sys.argv[2]
	eSizeTh = int(sys.argv[3])

	count = 0

	wccDict = getWccListDict(path, graph, eSizeTh)
	print('get wcc list done')
	for layer, wccList in wccDict.items():
		intersectionEdgeDict = getGraph(layer, wccList, path, graph)
		if intersectionEdgeDict:
			dumpEdges(layer, intersectionEdgeDict, graph)
			print(f'dump edges in layer {layer}, count {len(intersectionEdgeDict)}')
			count += len(intersectionEdgeDict)
	print(f'done, {count = }')
	with open(f'{graph}_datasetInfo.log', 'w') as f:
		f.write(f'{count = }')