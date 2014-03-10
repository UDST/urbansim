import numpy as np
from numpy.linalg import inv
from numpy.core.umath_tests import inner1d


def initialize_gpu():
    from cudamat import cudamat as cm
    global cm

    # this class wraps matrix operations so that you can switch between numpy
    # and cuda operations
    cm.cuda_set_device(0)
    cm.init()

# the keyword "inplace" for all of these functions re-uses the memory wherever
# possible use this if you don't want allocate more memory on the gpu which can
# be expensive, and don't need to reuse the matrix being transformed


def random(size, typ='numpy'):
    return PMAT(np.random.uniform(size=size).reshape(1, size))


class PMAT:

    def __init__(self, mat, typ='numpy'):
        self.typ = typ
        if (type(mat) != np.ndarray and
                type(mat) != np.matrix and
                type(mat) != np.float64):
            self.typ = 'cuda'
            self.mat = mat
        elif typ == 'numpy':
            self.mat = mat
        elif typ == 'cuda':
            self.mat = cm.CUDAMatrix(mat)

    def multiply(self, mat):
        if self.typ == 'numpy':
            return PMAT(np.dot(self.mat, mat.mat))
        elif self.typ == 'cuda':
            return PMAT(cm.dot(self.mat, mat.mat))

    def exp(self, inplace=False):
        if self.typ == 'numpy':
            return PMAT(np.exp(self.mat))
        elif self.typ == 'cuda':
            if inplace:
                self.mat = cm.exp(self.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                cm.exp(self.mat, target=target)
                return PMAT(target)

    def log(self, inplace=False):
        if self.typ == 'numpy':
            return PMAT(np.log(self.mat))
        elif self.typ == 'cuda':
            if inplace:
                self.mat = cm.log(self.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                cm.log(self.mat, target=target)
                return PMAT(target)

    # find the first positive value along an axis - for use in doing random
    # choice
    def firstpositive(self, axis):
        if self.typ == 'numpy':
            return PMAT(np.argmax((self.mat + 1.0).astype('i4'), axis=axis))

    def cumsum(self, axis):
        if self.typ == 'numpy':
            return PMAT(np.cumsum(self.mat, axis=axis))
        # elif self.typ == 'cuda':
        #  return PMAT(misc.cumsum(self.mat,axis=axis))

    def argmax(self, axis):
        if self.typ == 'numpy':
            return PMAT(np.argmax(self.mat, axis=axis))

    def transpose(self):
        if self.typ == 'numpy':
            return PMAT(np.transpose(self.mat))
        # WARNING, always in place
        elif self.typ == 'cuda':
            self.mat.transpose()

    def reshape(self, rowlen, collen):
        if rowlen == -1:
            rowlen = self.size() / collen
        if collen == -1:
            collen = self.size() / rowlen
        if self.typ == 'numpy':
            self.mat = np.reshape(self.mat, (rowlen, collen), order='F')
            return self
        # WARNING, always in place
        elif self.typ == 'cuda':
            self.mat = self.mat.reshape((rowlen, collen))
            return self

    def size(self):
        if self.typ == 'numpy':
            return self.mat.size
        elif self.typ == 'cuda':
            return self.mat.shape[0] * self.mat.shape[1]

    def sum(self, axis, shorten=0):
        if self.typ == 'numpy':
            # this is weird, but a numpy sum return flat array sometimes and
            # we want 2D matrices
            # return PMAT(np.sum(self.mat,axis=axis,dtype="float64"))
            if axis == 0:
                return PMAT(np.reshape(
                    np.sum(self.mat, axis=axis, dtype="float64"), (1, -1)))
            if axis == 1:
                return PMAT(np.reshape(
                    np.sum(self.mat, axis=axis, dtype="float64"), (-1, 1)))
            # return
            # PMAT(np.array(np.matrix(self.mat).sum(axis=axis,dtype="float64")))
        elif self.typ == 'cuda':
            return PMAT(self.mat.sum(axis=axis))

    def get_mat(self):
        if self.typ == 'numpy':
            return self.mat
        if self.typ == 'cuda':
            return self.mat.asarray()

    def shape(self):
        return self.mat.shape

    def subtract(self, mat, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat - mat.mat)
        if self.typ == 'cuda':
            if inplace:
                self.mat.subtract(mat.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.subtract(mat.mat, target=target)
                return PMAT(target)

    def divide_by_row(self, rowvec, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat / rowvec.mat)
        elif self.typ == 'cuda':
            if inplace:
                rowvec.mat.reciprocal()
                self.mat.mult_by_row(rowvec.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                rowvec.mat.reciprocal()
                self.mat.mult_by_row(rowvec.mat, target=target)
                return PMAT(target)

    def multiply_by_row(self, rowvec, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat * rowvec.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat.mult_by_row(rowvec.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.mult_by_row(rowvec.mat, target=target)
                return PMAT(target)

    def multiply_by_col(self, colvec, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat * colvec.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat.mult_by_col(colvec.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.mult_by_col(colvec.mat, target=target)
                return PMAT(target)

    def add_row_vec(self, rowvec, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat + rowvec.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat.add_row_vec(rowvec.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.add_row_vec(rowvec.mat, target=target)
                return PMAT(target)

    def add_col_vec(self, colvec, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat + colvec.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat.add_col_vec(colvec.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.add_col_vec(colvec.mat, target=target)
                return PMAT(target)

    def element_multiply(self, mat, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat * mat.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat = self.mat.mult(mat.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.mult(mat.mat, target=target)
                return PMAT(target)

    def element_add(self, mat, inplace=False):
        if self.typ == 'numpy':
            return PMAT(self.mat + mat.mat)
        elif self.typ == 'cuda':
            if inplace:
                self.mat = self.mat.add(mat.mat)
                return self
            else:
                target = cm.empty(self.mat.shape)
                self.mat.add(mat.mat, target=target)
                return PMAT(target)

    def clamptomin(self, val):
        assert self.typ == "numpy"
        self.mat[self.mat < val] = val

    def inftoval(self, val):
        assert self.typ == "numpy"
        self.mat[np.isinf(self.mat)] = val

    def nantoval(self, val):
        assert self.typ == "numpy"
        self.mat[np.isnan(self.mat)] = val

    def __str__(self):
        if self.typ == 'numpy':
            return str(self.mat)
        elif self.typ == 'cuda':
            return str(self.get_mat())
