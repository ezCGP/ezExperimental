
class Ting():
	def __init__(self):
		self.genome = [[0,0,0,0],[1,1,1,1],[2,2,2,2]]

	def __setitem__(self, index: tuple, value):
		self.genome[index[0]][index[1]] = value

t=Ting()
t[(1,2)]=50