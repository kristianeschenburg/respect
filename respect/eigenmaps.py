import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

from networkx import is_connected
from networkx import from_numpy_matrix

def eigenmaps(similarity, sinds, evecs):
    """
    Compute the eigenmaps of a given similarity matrix.

    Parameters:
    - - - - -
    similarity: float, array
        symmetric similarity matrix
    evecs: int
        number of eigenvectors to compute
    normalize: bool
        generate [0, 1] ranging eigenvectors

    Returns:
    - - - -
    vectors: float, array
        eigenvectors of similarity matrix
    """

    similarity[np.isnan(similarity)] = 0
    similarity[np.isinf(similarity)] = 0

    row_sums = (np.abs(similarity).sum(1) != 0)
    col_sums = (np.abs(similarity).sum(0) != 0)

    assert np.all(row_sums == col_sums)
    S = similarity[:, row_sums][col_sums, :]
    inds = sinds[row_sums]

    # Compute dissimilarity matrix
    distance = norm(S)

    # Remove strongest edges, up to point graph becomes disconnected
    adj = adjacency(distance)

    print('Computing laplacian.')
    W = np.multiply(adj, S)
    D = np.diag(np.sum(W, 0))
    L = np.subtract(D, W)

    print('Computing the dominant ' + str(evecs) + ' connectopic maps...')
    l, y = eigh(L, D, eigvals=(0, evecs))

    corr_vec = np.arange(len(inds))

    # compute sign-flipped eigenvectors
    sign_flipped = np.zeros((y.shape))
    for evec in range(1, y.shape[1]):
        temp = np.multiply(y[:, evec], 
                        np.sign(np.corrcoef(y[:, evec], corr_vec)[0, 1]))

        sign_flipped[:, evec] = temp

    sign_flipped[:, 0] = y[:, 0]

    signed = np.zeros((32492, sign_flipped.shape[1]-1))
    for evec in range(0, sign_flipped.shape[1]-1):
        signed[inds, evec] = sign_flipped[:, evec+1]

    return signed


def norm(X):

    """
    Comput dissimilarity matrix.

    Parameters:
    - - - - -
    X: float, array
        similarity matrix / data samples
    """

    return squareform(pdist(X))


def adjacency(X):

    """
    Threshold matrix, up to point where it is no longer connected

    Parameters:
    - - - - -
    X: float, array
        dissimilarity matrix
    """


    emin = 0
    emax = np.max(X)
    tol = 0.0001
    maxiter = 1000
    cntr = 0
    
    done = False
    while not done:        
        e = (emin + emax) / 2
        A = (X < e) - np.eye(X.shape[0])
        G = from_numpy_matrix(A)
        if is_connected(G):
            emax = e
            if (emax - emin) < tol:
                done = True
        else:
            emin = e       
        cntr += 1
        if cntr == maxiter:
            done = True      
    
    return A
