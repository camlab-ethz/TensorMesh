import sys 
sys.path.append("../..")


import torch 
import random
import scipy.sparse
from torch_fem.sparse import SparseMatrix

def test_block_coo_naive():
    edata = torch.stack([torch.ones([2,2]), torch.ones([2,2])*2],0)
    row   = torch.tensor([0, 1])
    col   = torch.tensor([0, 1])
    shape = (2,2)
    mat = SparseMatrix.from_block_coo(edata, row, col, shape)
    
    label = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2]
    ], dtype=torch.float)

    assert torch.allclose(mat.to_dense(), label)


def test_block_coo_random(n_times=4):
    for _ in range(n_times):
        block_size = random.randint(1, 10)
        r          = random.randint(1, 100)
        density    = random.random()
        scipy_mat  = scipy.sparse.random(r, r, density=density, format="coo")
        col, row   = scipy_mat.col, scipy_mat.row
        edata      = torch.rand([len(col), block_size, block_size])
        row        = torch.from_numpy(row)
        col        = torch.from_numpy(col)
        shape      = scipy_mat.shape
        mat        = SparseMatrix.from_block_coo(edata, row, col, shape)

        label = torch.zeros([r * block_size, r * block_size])
        for i, j, v in zip(row, col, edata):
            label[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] += v
        
        assert torch.allclose(mat.to_dense(), label)
