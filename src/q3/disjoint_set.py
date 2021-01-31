import numpy as np


# disjoint-set forests using union-by-rank and path compression (sort of).
class universe:
    def __init__(self, n_elements):
        self.num_set = n_elements
        self.num_vertice = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # parent

    def size(self, x):
        return self.elts[x, 1]

    def num_sets(self):
        return self.num_set

    def find(self, x):
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x, 2] = y
        return y

    def join(self, x, y):
        # x = int(x)
        # y = int(y)
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]
        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num_set -= 1

    def all_comp(self):
        ret = []
        i, n = 0, 0
        while n != self.num_set:
            if self.elts[i, 2] == i:
                ret.append(i)
                n += 1
            i += 1
        return ret
    
    def all_vertices_in_comp(self, x):
        root = self.find(x)
        pixs = []
        for i in range(self.num_vertice):
            if self.find(i) == root:
                pixs.append(i)
        return pixs

class vp_hash_table:
    def __init__(self, h, w):
        self.ht = np.zeros(shape=(h*w+w), dtype=tuple)
        self.h = h
        self.w = w
        for y in range(h):
            for x in range(w):
                self.ht[y*w+x] = (y,x)
    
    def vertice2pix(self, v):
        return self.ht[v]

    def pix2vertice(self, x, y):
        return y*self.w+x