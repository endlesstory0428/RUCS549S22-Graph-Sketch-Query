import numpy as np
import matplotlib.pyplot as plt

def addNoise(posList, scale = 1.41):
	for idx, stroke in enumerate(posList):
		noise = np.random.normal(scale = scale, size = stroke.shape).round().astype(np.int32)
		# print(noise)
		posList[idx] += noise
	return posList

class node(object):
	def __init__(self, idx = None):
		super(node, self).__init__()
		self.idx = np.random.randint(3) if idx is None else idx
		return

	def getPosList(self):
		stroke0 = [
			[
				[48, 12], [21, 12], [15, 24], [10, 44], [11, 62], [17, 75], [31, 78], [60, 79], [80, 66], [93, 48], [94, 31], [81, 17], [63, 11], [51, 10]
			],
			[
				[36, 15], [18, 37], [14, 67], [19, 79], [35, 90], [57, 90], [84, 74], [95, 60], [98, 51], [98, 37], [93, 30], [68, 22], [57, 18], [30, 18]
			],
			[
				[54, 19], [45, 19], [31, 24], [18, 45], [12, 62], [12, 69], [18, 79], [30, 84], [57, 84], [69, 80], [78, 68], [78, 65], [84, 56], [85, 47], [86, 23], [85, 18], [62, 12], [51, 9], [35, 6]
			]
		]
		posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0]]
		region = [
			[
				[22, 21], [75, 64]
			],
			[
				[27, 27], [85, 67]
			],
			[
				[29, 32], [72, 69]
			]
		]
		return addNoise(posList), region[self.idx]

class link(object):
	def __init__(self, src, tgt):
		super(link, self).__init__()
		self.src = np.array(src, dtype = np.int32)
		self.tgt = np.array(tgt, dtype = np.int32)
		return

	def getPosList(self):
		dist = np.sqrt(np.sum(np.square(self.src - self.tgt)))
		samples = dist // 50 + 2
		stroke0 = np.linspace(self.src, self.tgt, int(samples)).round().astype(np.int32)
		posList = np.array([stroke0], dtype = np.int32)
		return addNoise(posList)

class label(object):
	def __init__(self, char, idx = None):
		super(label, self).__init__()
		self.char = char
		self.idx = np.random.randint(2) if idx is None else idx
		return

	def getPosList(self):
		if self.char == 'F':
			stroke0 = [
				[
					[2, 3], [15, 2]
				],
				[
					[6, 2], [27, 2]
				]
			]

			stroke1 = [
				[
					[5, 3], [5, 22], [5, 40]
				],
				[
					[6, 2], [6, 13], [4, 21], [4, 30], [2, 39]
				]
			]

			stroke2 = [
				[
					[5, 21], [9, 17], [18, 17]
				],
				[
					[3, 16], [23, 16]
				]
			]

			region = [
				[
					[0, 0], [18, 40]
				],
				[
					[0, 0], [26, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1, stroke2]]
			return addNoise(posList, 0.707), region[self.idx]


		if self.char == 'K':
			stroke0 = [
				[
					[3, 4], [1, 37]
				],
				[
					[20, 5], [3, 19], [21, 36]
				]
			]

			stroke1 = [
				[
					[18, 4], [2, 23], [23, 38]
				],
				[
					[1, 5], [4, 35]
				]
			]

			region = [
				[
					[0, 0], [23, 40]
				],
				[
					[0, 0], [22, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]


		if self.char == '0':
			stroke0 = [
				[
					[9, 1], [6, 2], [3, 13], [1, 28], [3, 34], [7, 38], [17, 38], [22, 33], [25, 19], [26, 14], [20, 6], [14, 4], [6, 3]
				],
				[
					[6, 10], [5, 15], [5, 28], [9, 36], [15, 36], [17, 31], [20, 25], [20, 10], [13, 5], [5, 4], [2, 7]
				]
			]

			region = [
				[
					[0, 0], [27, 40]
				],
				[
					[0, 0], [21, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '1':
			stroke0 = [
				[
					[5, 2], [2, 38]
				],
				[
					[2, 10], [9, 4], [8, 37]
				]
			]

			stroke1 = [
				[
				],
				[
					[2, 37], [13, 37]
				]
			]

			region = [
				[
					[0, 0], [6, 40]
				],
				[
					[0, 0], [15, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '2':
			stroke0 = [
				[
					[3, 7], [9, 2], [14, 3], [17, 6], [17, 17], [14, 24], [10, 33], [2, 36], [20, 33]
				],
				[
					[8, 2], [14, 4], [17, 8], [17, 20], [14, 26], [3, 34], [19, 36]
				]
			]

			region = [
				[
					[0, 0], [22, 40]
				],
				[
					[0, 0], [21, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '3':
			stroke0 = [
				[
					[5, 4], [7, 3], [13, 3], [15, 5], [15, 13], [8, 19], [14, 19], [19, 23], [19, 26], [9, 37], [5, 38], [2, 35]
				],
				[
					[12, 1], [18, 2], [21, 7], [21, 15], [18, 19], [11, 20], [21, 21], [22, 31], [17, 38], [12, 38], [2, 33]
				]
			]

			region = [
				[
					[0, 0], [20, 40]
				],
				[
					[0, 0], [23, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '4':
			stroke0 = [
				[
					[5, 1], [2, 19], [19, 19]
				],
				[
					[3, 5], [3, 16], [13, 16], [18, 13]
				]
			]

			stroke1 = [
				[
					[13, 1], [1, 39]
				],
				[
					[11, 2], [9, 6], [7, 39]
				]
			]

			region = [
				[
					[0, 0], [19, 40]
				],
				[
					[0, 0], [19, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '5':
			stroke0 = [
				[
					[9, 3], [20, 3]
				],
				[
					[6, 5], [6, 18], [13, 19], [18, 26], [18, 31], [14, 35], [2, 33]
				]
			]

			stroke1 = [
				[
					[9, 2], [6, 15], [13, 17], [24, 27], [24, 31], [15, 36], [6, 36], [1, 32]
				],
				[
					[1, 8], [13, 6]
				]
			]

			region = [
				[
					[0, 0], [26, 40]
				],
				[
					[0, 0], [19, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '6':
			stroke0 = [
				[
					[9, 1], [6, 5], [3, 13], [1, 35], [4, 39], [10, 38], [14, 32], [14, 26], [8, 21], [2, 21]
				],
				[
					[6, 3], [1, 24], [1, 31], [9, 35], [15, 35], [15, 33], [2, 24]
				]
			]

			region = [
				[
					[0, 0], [15, 40]
				],
				[
					[0, 0], [15, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '7':
			stroke0 = [
				[
					[2, 4], [18, 2], [7, 39]
				],
				[
					[3, 1], [14, 2], [6, 38]
				]
			]

			stroke1 = [
				[
				],
				[
					[1, 20], [17, 20]
				]
			]

			region = [
				[
					[0, 0], [18, 40]
				],
				[
					[0, 0], [17, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0, stroke1] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '8':
			stroke0 = [
				[
					[9, 1], [1, 8], [1, 12], [15, 25], [15, 34], [12, 39], [3, 38], [2, 35], [14, 7], [12, 2]
				],
				[
					[12, 1], [4, 9], [19, 27], [19, 31], [7, 38], [2, 36], [2, 31], [18, 6], [18, 1]
				]
			]

			region = [
				[
					[0, 0], [16, 40]
				],
				[
					[0, 0], [20, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

		if self.char == '9':
			stroke0 = [
				[
					[16, 4], [9, 1], [1, 10], [1, 18], [9, 18], [15, 8], [10, 38]
				],
				[
					[18, 7], [15, 2], [8, 1], [1, 7], [3, 12], [9, 13], [18, 7], [6, 39]
				]
			]

			region = [
				[
					[0, 0], [16, 40]
				],
				[
					[0, 0], [19, 40]
				]
			]
			posList = [np.array(stroke[self.idx], dtype = np.int32) for stroke in [stroke0] if stroke[self.idx]]
			return addNoise(posList, 0.707), region[self.idx]

if __name__ == '__main__':
	# nodeTemplate = node(2)
	# posList, region = nodeTemplate.getPosList()
	# linkTemplate = link((64, 77), (206, 453))
	# posList = linkTemplate.getPosList()
	labelTemplate = label('K', 1)
	posList, region = labelTemplate.getPosList()
	for stroke in posList:
		plt.plot(stroke[:, 0], -stroke[:, 1])
	plt.show()