import numpy as np
from respect.eigenmaps import eigenmap

class RecursivePartitioning(object):
    
    """
    Applying recursive, binary spectral partitioning of a similarity matrix.
    
    Parameters:
    - - - - -
    max_depth: int
        maximum number of binary splits to perform
    evecs: int
        number of eigenvectors to computer at each iteration
        
    Returns:
    - - - -
    clusters: int, array
        array of size (N x max_depth), where N is the number of data
        samples in the original similarity matrix
    """
    
    def __init__(self, max_depth=2, evecs=6):
        
        """
        Instantiate RecursivePartitioning object.
        """

        self.max_depth = max_depth
        self.evecs = evecs
    
    def fit(self, D):
        
        """
        Method to fit the recursive partitions.
        
        Parameters:
        - - - - -
        D: float, array
            similarity matrix
        """
        
        print('Fitting clusters with maximum depth of %i\nusing %i eigenvectors per split.\n' % (self.max_depth, self.evecs))
        
        inds = np.arange(D.shape[0])

        print('Recursion step.')
        [clust, idx, depths, evx] = self._split(D, inds, 
                                                     [], [], [], [], 0)

        print('Aggregation step.')
        [clusters, eigenvectors] = self._combine(clust, idx, depths, evx)
        
        self.clusters_ = clusters
        self.eigenvectors_ = eigenvectors
        
    def _split(self, S, indices, 
                     init_clusters, init_inds, init_depths, init_evecs, 
                     current_depth):
        
        """
        Private recursive function to perform mult-level binary spectral clustering.
        
        Parameters:
        - - - - -
        S: float, array
            similarity matrix
        indices: int, list
            initial list of indices corresponding to indices of samples
        init_clusters, init_inds, init_depths: lists
            keep track of Fiedler vector, split direction, and partition size
            at each depth
        current_depth: int
            current recursion depth
        """
        
        if current_depth > self.max_depth:
                
            return False
    
        else:

            # compute eigenmaps of current similarity matrix
            y = eigenmap(S, self.evecs)

            # get fielder vector
            f = y[:, 1]

            # get positive and negative indices of fielder vector
            pi = np.where(f > 0)[0]
            pix = indices[pi]
            
            ni = np.where(f < 0)[0]
            nix = indices[ni]

            # get positive and negative sub-similarity matrices
            # each of these are then processed by eigendepth
            pd = S[:, pi][pi, :]
            nd = S[:, ni][ni, :]

            # recursive step down each positive / negative brach

            init_clusters.append(f)
            init_inds.append(pix)
            init_inds.append(nix)
            
            init_depths.append(current_depth)
            py = eigenmap(pd, self.evecs)
            py = self._signcorrect(py, pix)
            py = (py - py.min(0)) / (py.max(0) - py.min(0))
            
            ny = eigenmap(nd, self.evecs)
            ny = self._signcorrect(ny, nix)
            ny = (ny - ny.min(0)) / (ny.max(0) - ny.min(0))

            # left branch
            init_evecs.append(py[:, 1:])
            init_evecs.append(ny[:, 1:])
            
            g = self._split(pd, pix, 
                           init_clusters, init_inds, init_depths, init_evecs, 
                           current_depth+1)
            
            # right branch
            g = self._split(nd, nix, 
                           init_clusters, init_inds, init_depths, init_evecs, 
                           current_depth+1)
        
        return [init_clusters, init_inds, init_depths, init_evecs]
    

    def _combine(self, clusters, indices, depths, evecs):
        
        """
        Aggregate the results of spectral partitioning.
        """
        
        ns = clusters[0].shape[0]
        niters = len(clusters)
        
        z = np.zeros((ns, self.max_depth+1))
        e = np.zeros((ns, self.max_depth+1, self.evecs))
        depth_map = {k: 1 for k in np.arange(self.max_depth+1)}
        
        for j in np.arange(niters):
            
            cdepth = depths[j]
            
            if cdepth <= self.max_depth:
                
                F = np.zeros((clusters[j].shape))
                F[clusters[j]>0] = depth_map[cdepth]
                F[clusters[j]<0] = depth_map[cdepth]+1
                
                pinds = np.where(clusters[j] > 0)[0]
                ninds = np.where(clusters[j] < 0)[0]

                linds = indices[j*2]
                rinds = indices[j*2 + 1]
                
                epy = evecs[j*2]
                eny = evecs[j*2 + 1]

                z[linds, cdepth] = F[pinds]
                z[rinds, cdepth] = F[ninds]
                
                e[linds, cdepth, :] = epy
                e[rinds, cdepth, :] = eny

                depth_map[cdepth] += 2
        
        return [z, e]
    
    def _signcorrect(self, vectors, indices):
        
        corr_vec = np.arange(len(indices))
        for evec in range(1, vectors.shape[1]):
            vectors[:, evec] = np.multiply(vectors[:, evec],
                np.sign(np.corrcoef(vectors[:, evec], corr_vec)[0, 1]))
        
        return vectors