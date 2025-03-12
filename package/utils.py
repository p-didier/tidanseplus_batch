import time, copy
import numpy as np
import networkx as nx
from .meta_utils import *
import scipy.linalg as sla
import scipy.signal as ssig
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class Signal:
    """Signal class for a single node."""
    td: dict[str, np.ndarray] = field(default_factory=dict) # time-domain signals dictionary
    tdFused: dict[str, np.ndarray] = field(default_factory=dict) # time-domain signals dictionary (fused)
    tdTilde: dict[str, np.ndarray] = field(default_factory=dict) # time-domain signals dictionary (observation vectors)
    wd: dict[str, np.ndarray] = field(default_factory=dict) # WOLA-domain signals dictionary
    wdFused: dict[str, np.ndarray] = field(default_factory=dict) # WOLA-domain signals dictionary (fused)
    wdTilde: dict[str, np.ndarray] = field(default_factory=dict) # WOLA-domain signals dictionary (observation vectors)

    def compute_stfts(self, cfg: 'Parameters'):
        """Compute the STFTs of the signals."""
        for key, val in self.td.items():
            self.wd[key] = cfg.get_stft(val)
            if cfg.singleBin is not None:
                self.wd[key] = self.wd[key][[cfg.singleBin], :, :]
    
    def fuse(self, Pk: np.ndarray):
        """Fuse the signals."""
        for key, val in self.wd.items():
            self.wdFused[key] = inner(Pk.conj(), val)
        for key, val in self.td.items():
            self.tdFused[key] = inner(Pk.conj(), val)

    def extract(self, idx1: int, idx2: int):
        """Return a copy of the class instance containing only the signals
        from mic `idx1` to mic `idx2`.
        NB: does not extract anything from the observation vectors (tilde)."""
        out = copy.deepcopy(self)
        for key, val in self.td.items():
            out.td[key] = val[:, idx1:idx2, :]
        for key, val in self.wd.items():
            out.wd[key] = val[:, idx1:idx2, :]
        return out


@dataclass
class SignalsCollection:
    sigs: list[Signal] = field(default_factory=list)

    def single_bin(self, idx: int):
        """Return a copy of the class instance containing only the signals
        from the single frequency bin `idx`."""
        out = copy.deepcopy(self)
        for s in out.sigs:
            for key, val in s.wd.items():
                s.wd[key] = val[[idx], :, :]
            for key, val in s.wdFused.items():
                s.wdFused[key] = val[[idx], :, :]
            for key, val in s.wdTilde.items():
                s.wdTilde[key] = val[[idx], :, :]
        return out

    def compute_stfts(self, cfg: 'Parameters'):
        """Compute the STFTs of the signals."""
        for s in self.sigs:
            s.compute_stfts(cfg)
    
    def fuse(self, Pk: list):
        """Fuse the signals from each node."""
        for k, s in enumerate(self.sigs):
            s.fuse(Pk[k])
    
    def centralize(self) -> Signal:
        """Centralize the signals from each node into a global vector."""
        sigVect = Signal(
            td={key: np.concatenate([s.td[key] for s in self.sigs], axis=0)  # channel axis: 0
                for key in self.sigs[0].td.keys()},
            wd={key: np.concatenate([s.wd[key] for s in self.sigs], axis=1)  # channel axis: 1
                for key in self.sigs[0].wd.keys()}
        )
        return sigVect
    
    def build_obs_vectors(
            self,
            algo: str='DANSE',
            k: int=None,
            wasnsPruned: dict[int, 'WASN']=None,
        ) -> np.ndarray:
        """
        Build the observation vectors.
        
        Parameters
        ----------
        algo : str
            Algorithm to build the observation vectors for.
        k : int
            Updating node index (used for TI-DANSE+ only).
        wasnsPruned : dict[int, WASN]
            Wireless acoustic sensor network objects, pruned to trees per root
            node index (used for TI-DANSE+ only).
        """

        nNodes = len(self.sigs)

        def _process(q, z, _lSet=None, _wasn: WASN=None):
            """Helper function which constructs a tuple of elements,
            each representing the non-local entries in the network-wide
            filters (DANSE, TI-DANSE, and TI-DANSE+) at node `q`."""
            if algo == 'DANSE':
                t = tuple(z[m] for m in range(nNodes) if m != q)
            elif algo == 'TI-DANSE':
                t = (np.sum(  # eta_{-q} (in-network sum of fused signals)
                    np.array([z[m] for m in range(nNodes) if m != q]),
                    axis=0
                ),)
            elif algo == 'TI-DANSE+':
                t = tuple(   # eta_{-> q} (multiple partial in-network sums)
                    np.sum(
                        np.array([z[m] for m in range(nNodes) if _wasn.lq[m] == l]),
                        axis=0
                    ) for l in _lSet
                )
            return t

        lSet, wasn = None, None  # default for other algorithms than TI-DANSE+
        for q, s in enumerate(self.sigs):
            if algo == 'TI-DANSE+':
                if q != k:
                    continue  # only compute observation vector yTilde at root node
                    # `k` in TI-DANSE+ (the other nodes do not have access to the
                    # necessary data)
                wasn = wasnsPruned[q]
                lSet = np.where(wasn.adjacency[q, :] == 1)[0]
            
            # WOLA-domain signals
            for typ, y in s.wd.items():
                z = [self.sigs[m].wdFused[typ] for m in range(len(self.sigs))]  # fused signals alias
                t = _process(q, z, lSet, wasn)
                self.sigs[q].wdTilde[typ] = np.concatenate((y,) + t, axis=1)
            # Time-domain signals
            for typ, y in s.td.items():
                z = [self.sigs[m].tdFused[typ] for m in range(len(self.sigs))]  # fused signals alias
                t = _process(q, z, lSet, wasn)
                self.sigs[q].tdTilde[typ] = np.concatenate((y,) + t, axis=0)

    def get_tilde(self, ref: str, td: bool=False):
        """Returns list of tilde signals from all nodes."""
        if td:
            return [s.tdTilde[ref] for s in self.sigs]
        return [s.wdTilde[ref] for s in self.sigs]
    
    def get(self, ref: str, td: bool=False):
        """Returns list of signals from all nodes."""
        if td:
            return [s.td[ref] for s in self.sigs]
        return [s.wd[ref] for s in self.sigs]
    
    def get_frame(self, l: int):
        """Returns new `SignalsCollection` object with just data
        from frame `l` (only keeping WOLA-domain signals)."""
        return SignalsCollection(
            sigs=[Signal(
                wd={key: s.wd[key][..., [l]] for key in s.wd.keys()}
            ) for s in self.sigs]
        )


@dataclass
class Parameters:
    # Room
    c: float = 343  # speed of sound [m/s]
    fs: int = 16e3  # sampling frequency [Hz]
    rd: float = 5  # room dimension [m]
    dim: int = 2  # room dimensionality (2D or 3D)
    t60: float = 0.  # reverberation time [s]
    nGlobDes: int = 1  # number of global desired sources
    nGlobNoi: int = 1  # number of global noise sources
    sourceSignalType: str = 'random'   # latent source signals type: 'random' if stochastic or 'from_file' if deterministic
    globDesSourcesFiles: np.ndarray = field(default_factory=lambda: np.array([]))  # array of paths to global latent desired source signal files (used iff `sourceSignalType == 'from_file'`)`
    globNoiSourcesFiles: np.ndarray = field(default_factory=lambda: np.array([]))  # array of paths to global latent noise source signal files (used iff `sourceSignalType == 'from_file'`)`
    
    # Network
    K: int = 10  # number of nodes
    Mk: int = 3  # number of sensors per node
    mindWalls: float = 0  # minimum distance between nodes and walls [m]
    mindSourceSensor: float = 0.5  # minimum distance between sources and sensors [m]
    d: float = 0.2  # array element separation [m]
    commDist: float = 1.5  # communication distance [m] (used at ad-hoc topology initialization, but `connectivity` plays a role)
    connectivity: np.ndarray = field(default_factory=lambda: np.array([]))  # amount of ad-hoc topology connectivity
    selfNoiseFactor: float = 0.1  # as a %tage of the mixed desired signal as observed by the first microphone, at each node (0 for no self-noise)

    # General processing parameters
    useiDANSEifPossible: bool = True  # if true, use the iterationless version of DANSE if S_LOCAL > 0 (ref. [1], cf. `main.py`).
    TmaxOnlineMode: float = 10.  # [s] max. signal duration, only for online-mode processing
    useRss: bool = True  # if True, use Rss instead of Ryy-Rnn

    # Hyperparameters
    nMCbatch: int = 10  # number of Monte Carlo runs for batch processing
    algNames: list[str] = field(default_factory=lambda: ['Centr.', 'DANSE', 'TI-DANSE', 'TI-DANSE+'])  # algorithms to run

    # Algorithmic parameters
    Q: int = 1  # DANSE dimension
    nPosFreqs: int = None  # number of positive frequencies (None for all, initialized in `__post_init__`)

    # STFT
    ndft: int = 512  # FFT size
    hop: int = 256  # hop size
    win: str = 'sqrthann'  # window type

    pruningType: str = 'minst'  
    # >>> 'minst': minimum spanning tree
    # >>> 'maxst': maximum spanning tree
    # >>> 'mmut': multiple "max-branch" pruning, maximizing the # of branches at root node (different at every iteration)

    ############################################
    singleBin: int = None  # if not None, use only a single frequency bin (index `singleBin`) in WOLA.
    ############################################
    seed: int = 42
    exportFolder: str = f'out/at_{time.strftime("%Y%m%d_%H%M%S")}'
    exportFolderSuffix: str = ''

    def __post_init__(self):
        self.M = self.Mk * self.K
        self.Dk = self.Mk + self.Q * (self.K - 1)
        self.DkTI = self.Mk + self.Q
        self.nPosFreqs = self.ndft // 2 + 1 if self.singleBin is None else 1

    def dump_to_yaml(self, path=None):
        dump_to_yaml_template(self, path)

    def load_from_yaml(self, path: str):
        return load_from_yaml(path, self)

    def load_from_dict(self, d: dict):
        return load_from_dict(d, self)
    
    def dump_to_txt(self, path=None):
        with open(path, 'w') as f:
            f.write(self.__repr__())
    
    def get_stft(self, x):
        """Compute the STFT of the signal `x`."""
        stft: np.ndarray = ssig.stft(
            x,
            fs=self.fs,
            nperseg=self.ndft,
            noverlap=self.ndft - self.hop,
            window=self.win,
            return_onesided=True,
        )[2]
        if stft.ndim == 3:
            return np.moveaxis(stft, 1, 0)
        else:   # single channel
            return stft
        
    def __repr__(self):
        """Return a formatted string representation of the class."""
        return '\n'.join([
            f'{key}: {val}' if not isinstance(val, (list, dict)) 
            else f'{key}: {type(val).__name__} with {len(val)} items: {val}'
            for key, val in self.__dict__.items()
        ]) + '\n' + '-' * 25 + ' ^^^ Parameters ^^^ ' + '-' * 25 + '\n'


@dataclass
class WASN:
    nodePositions: np.ndarray
    sensorPositions: np.ndarray
    globalDesiredPositions: np.ndarray  # positions of global desired sources
    globalNoisePositions: np.ndarray   # positions of global noise sources
    adjacency: np.ndarray  # adjacency matrix
    commDist: float  # communication distance
    root: int = None  # root node for tree pruning
    lq: list = None  # list of children of `root` that are (grand-)parents of `k`
    rirs: list[list] = None  # room impulse responses (RIRs) [per sensor[per source]] [if using ISM]
    latent: 'LatentSignals' = None  # latent signals [per source] (quotes because forward referencing of class LatentSignals)
    # ^^^ sources order: global desired, global noise, local desired for each node, local noise for each node
    cfg: Parameters = None

    def plot(self, show=True):
        """Plots the WASN."""

        def _side_plot(ax, indices=[0,1]):
            ax.scatter(
                self.sensorPositions[:, :, indices[0]],
                self.sensorPositions[:, :, indices[1]],
                c='b', marker='x', label='Sensors'
            )
            # Plot circle around nodes to encompass sensors
            for k in range(self.cfg.K):
                circle = plt.Circle(
                    (
                        self.nodePositions[k, indices[0]],
                        self.nodePositions[k, indices[1]]
                    ),
                    self.cfg.d,
                    color='r',
                    fill=False
                )
                ax.add_artist(circle)
                # Add node number
                ax.text(
                    self.nodePositions[k, indices[0]] + 0.1,
                    self.nodePositions[k, indices[1]] + 0.1,
                    f'{k}',
                    fontsize=8
                )
                if self.root is not None:
                    # Increase line width of circle for root node
                    if k == self.root:
                        circle.set_linewidth(2)
            # Plot sources
            ax.scatter(
                self.globalDesiredPositions[:, indices[0]],
                self.globalDesiredPositions[:, indices[1]],
                c='g', marker='d', label='Global desired'
            )
            ax.scatter(
                self.globalNoisePositions[:, indices[0]],
                self.globalNoisePositions[:, indices[1]],
                c='k', marker='+', label='Global noise'
            )
            # Add dashed lines between nodes that communicate
            for k in range(self.cfg.K):
                for q in range(self.cfg.K):
                    if k > q and self.adjacency[k, q] == 1:
                        ax.plot([
                            self.sensorPositions[k, 0, indices[0]],
                            self.sensorPositions[q, 0, indices[0]]
                        ], [
                            self.sensorPositions[k, 0, indices[1]],
                            self.sensorPositions[q, 0, indices[1]]
                            ], 'k--')
            ax.set_xlim([0, self.cfg.rd])
            ax.set_ylim([0, self.cfg.rd])
            ax.set_aspect('equal', adjustable='box')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid('on')

        if self.cfg.dim == 3:
            fig = plt.figure()
            fig.set_size_inches(10.5, 3.5)
            ax = fig.add_subplot(131)
            # First subplot - top view
            _side_plot(ax, indices=[0, 1])
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title('Top view')
            # Second subplot - side view
            ax = fig.add_subplot(132)
            _side_plot(ax, indices=[0, 2])
            ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_title('Side view')
            # Third subplot - 3D view
            ax = fig.add_subplot(133, projection='3d')
            ax.scatter(
                self.globalDesiredPositions[:, 0],
                self.globalDesiredPositions[:, 1],
                self.globalDesiredPositions[:, 2],
                c='g', marker='d'
            )
            ax.scatter(
                self.globalNoisePositions[:, 0],
                self.globalNoisePositions[:, 1],
                self.globalNoisePositions[:, 2],
                c='k', marker='+'
            )
            for k in range(self.cfg.K):
                ax.scatter(
                    self.sensorPositions[k, :, 0],
                    self.sensorPositions[k, :, 1],
                    self.sensorPositions[k, :, 2],
                    c='b', marker='x'
                )
                ax.text(
                    self.nodePositions[k, 0],
                    self.nodePositions[k, 1],
                    self.nodePositions[k, 2],
                    f'{k}',
                    fontsize=8
                )
                # Add transparent sphere around nodes to encompass sensors
                if k != self.root:
                    ax.scatter(
                        self.nodePositions[k, 0],
                        self.nodePositions[k, 1],
                        self.nodePositions[k, 2],
                        c='r', marker='o', alpha=0.5,
                        s=100
                    )
                else:
                    ax.scatter(
                        self.nodePositions[self.root, 0],
                        self.nodePositions[self.root, 1],
                        self.nodePositions[self.root, 2],
                        c='y', marker='o', alpha=0.5,
                        s=100
                    )

                for k in range(self.cfg.K):
                    for q in range(self.cfg.K):
                        if k > q and self.adjacency[k, q] == 1:
                            ax.plot([
                                self.sensorPositions[k, 0, 0],
                                self.sensorPositions[q, 0, 0]
                            ], [
                                self.sensorPositions[k, 0, 1],
                                self.sensorPositions[q, 0, 1]
                            ], [
                                self.sensorPositions[k, 0, 2],
                                self.sensorPositions[q, 0, 2]
                            ], 'k--')
                
            fig.tight_layout()
            if show:
                plt.show(block=False)

        elif self.cfg.dim == 2:
            # 2D-only plot
            fig = plt.figure()
            fig.set_size_inches(5.5, 3.5)
            ax = fig.add_subplot(111)
            _side_plot(ax)
            fig.tight_layout()
            if show:
                plt.show(block=False)

        return fig

    def prune_tree(self, u):
        """Prune the WASN to a tree rooted at node `u`."""
        # Generate NetworkX graph
        Gnx: nx.Graph = nx.from_numpy_array(self.adjacency)
        # Get node positions 
        nodesPos = dict(
            [(k, self.sensorPositions[k, ...]) for k in range(self.cfg.K)]
        )
        
        # Add edge weights based on inter-node distance ((TODO -- is that a correct approach?))
        if nodesPos[0] is not None:
            for e in Gnx.edges():
                weight = np.linalg.norm(nodesPos[e[0]] - nodesPos[e[1]])
                Gnx[e[0]][e[1]]['weight'] = weight
        else:
            for e in Gnx.edges():
                Gnx[e[0]][e[1]]['weight'] = 1  # not "true-room" scenario

        if self.cfg.pruningType == 'minst':
            # ------------ Prune to minimum spanning tree ------------
            # Compute minimum spanning tree
            prunedWasnNX: nx.Graph = nx.minimum_spanning_tree(
                Gnx,
                weight='weight',
                algorithm='kruskal'
            )
        elif self.cfg.pruningType == 'maxst':
            # ------------ Prune to single tree with max. number of branches ------------
            # Compute maximum spanning tree
            prunedWasnNX: nx.Graph = nx.maximum_spanning_tree(
                Gnx,
                weight='weight',
                algorithm='kruskal'
            )
        elif self.cfg.pruningType == 'mmut':
            # ------------ Prune to tree with max. number of branches at root node ------------
            prunedWasnNX = mmut_pruning(Gnx, u)
        
        adjMat = nx.adjacency_matrix(prunedWasnNX).toarray()
        adjMat[adjMat > 0] = 1

        wasnPruned = copy.deepcopy(self)
        wasnPruned.adjacency = adjMat
        wasnPruned.root = u

        # Get list of `lq`'s, i.e., children of `u` that are (grand-)parents of `k`
        lq = [None for _ in range(self.cfg.K)]
        for k in range(self.cfg.K):
            if k != u:
                path = nx.shortest_path(prunedWasnNX, source=k, target=u)
                if len(path) == 2:
                    # `k` is directly connected to `u`
                    lq[k] = k  # direct neighbor
                else:
                    lq[k] = path[-2]  # (grand-)parent of `k`
        wasnPruned.lq = np.array(lq)

        return wasnPruned

    def randomize_adjmat(self):
        """Randomize adjacency matrix while ensuring connectivity."""
        def _gen_random_adjmat():
            """Generate random adjacency matrix."""
            adjMat = np.zeros((self.cfg.K, self.cfg.K))
            for k in range(self.cfg.K):
                for q in range(self.cfg.K):
                    if k > q:
                        adjMat[k, q] = adjMat[q, k] = 1 if np.random.choice([True, False]) else 0
            return adjMat
        # Ensure connectivity
        adjMat = _gen_random_adjmat()
        Gnx: nx.Graph = nx.from_numpy_array(adjMat)
        while not nx.is_connected(Gnx):
            adjMat = _gen_random_adjmat()
            Gnx = nx.from_numpy_array(adjMat)
        # Assign new adjacency matrix
        self.adjacency = adjMat

    def compute_lset(self, k: int):
        """Returns the set of neighbor indices of the root node, when the 
        ad-hoc WASN is pruned to a tree with node `k`."""
        wasnPruned = self.prune_tree(k)
        lSet = np.where(wasnPruned.adjacency[k, :] == 1)[0]
        return lSet
    
    def get_wet_signals(self):
        """Compute the wet (room-affected) microphone signals."""
        n = int(np.ceil(self.cfg.fs * self.cfg.TmaxOnlineMode))
        # Apply ISM-generated RIRs to latent signals
        print('Applying RIRs to latent signals...')
        return self.apply_rirs(n)

    def apply_rirs(self, n):
        """Apply RIRs to the time-domain latent signals."""
        s_g, n_g = [], []
        for k in range(self.cfg.K):
            s_gk = np.zeros((self.cfg.Mk, n))
            n_gk = np.zeros((self.cfg.Mk, n))
            for m in range(self.cfg.Mk):
                s_gk_curr = np.zeros(n)
                n_gk_curr = np.zeros(n)
                currRirs = self.rirs[k * self.cfg.Mk + m]
                for s in range(self.cfg.nGlobDes):
                    rir = currRirs[s]
                    latSig = self.latent.globalDesired[s]
                    s_gk_curr += apply_rir(rir, latSig)  # add up contributions from all global desired sources
                for s in range(self.cfg.nGlobNoi):
                    rir = currRirs[self.cfg.nGlobDes + s]
                    latSig = self.latent.globalNoise[s]
                    n_gk_curr += apply_rir(rir, latSig)
                s_gk[m, :] = s_gk_curr
                n_gk[m, :] = n_gk_curr
            s_g.append(s_gk)
            n_g.append(n_gk)
        
        return s_g, n_g


def apply_rir(rir, latSig):
    """Apply RIR to the time-domain latent signal."""
    wetSig = ssig.fftconvolve(rir, latSig)
    # wetSig = latSig * np.amax(np.abs(rir))
    return wetSig[:len(latSig)]


def mmut_pruning(graph: nx.Graph, v0):
    """
    Prune the graph to a tree with maximum number of branches at the root node.
    """
    
    # Step 1: Identify the fixed edges E_f
    E_f = [
        (u, v, data['weight'])
        for u, v, data in graph.edges(data=True) if u == v0 or v == v0
    ]
    
    # Step 2: Create a subgraph with just the fixed edges
    F = nx.Graph()
    F.add_weighted_edges_from(E_f)
    
    # Step 3: Initialize the MST with the fixed edges subgraph
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes)
    mst.add_edges_from(F.edges(data=True))
    
    # Step 4: Sort all edges by weight
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    
    # Step 5: Add remaining edges ensuring no cycles (using Kruskal's algorithm logic)
    for u, v, data in sorted_edges:
        if mst.has_edge(u, v):
            continue  # Skip fixed edges already in MST
        if not nx.has_path(mst, u, v):
            mst.add_edge(u, v, **data)
    
    return mst


def dot_to_p(s: float):
    return str(s).replace('.', 'p')


def myrand(size, rng=np.random.RandomState(), complex=False):
    """
    --a : int - number of rows
    --b : int - number of columns
    --rng : np.random.RandomState - random number generator
    """
    s = rng.uniform(-0.5, 0.5, size)
    if complex:
        return s + 1j * rng.uniform(-0.5, 0.5, size)
    else:
        return s


@dataclass
class Runner:
    cfg: Parameters
    Ryyct: list = field(default_factory=list)  # Ryy centralized theoretical SCM
    Rssct: list = field(default_factory=list)  # Rss centralized theoretical SCM
    Rnnct: list = field(default_factory=list)  # Rnn centralized theoretical SCM

    def get_nw_filters(
            self, Pk, w, PkTI, wTI,
            PkTIplus, wPlus=None, Tk=None,
            wasn: WASN=None
        ):
        """
        Compute all network-wide filters (DANSE, TI-DANSE, TI-DANSE+).

        Parameters:
        - Pk (list): List of matrices representing the filters for each neighbor node.
        - w (list): List of matrices representing the weights for each neighbor node.
        - PkTI (list): List of matrices representing the TI filters for each neighbor node.
        - wTI (list): List of matrices representing the TI weights for each neighbor node.
        - PkTIplus (list): List of matrices representing the TI-DANSE+ filters for each neighbor node.
        - wTIplus (list, optional): List of matrices representing the TI-DANSE+ weights for each neighbor node. Defaults to None.
        - Tk (list, optional): List of matrices representing the Tk matrices. Defaults to None.
        - wasn (WASN, optional): WASN object representing the pruned WASN. Defaults to None.

        Returns:
        - wNetWide (list): List of matrices representing the network-wide filters for each node.
        - wNetWideTI (list): List of matrices representing the TI network-wide filters for each node.
        - wNetWideTIplus (list): List of matrices representing the TI-DANSE+ network-wide filters for each node.
        """
        return self.get_nw_filters_indiv_algo(
            Pk, w, 'DANSE'
        ), self.get_nw_filters_indiv_algo(
            PkTI, wTI, 'TI-DANSE'
        ), self.get_nw_filters_indiv_algo(
            PkTIplus, wPlus, 'TI-DANSE+', Tk
        )

    def get_nw_filters_indiv_algo(
            self, Pk, w, algo='DANSE', Tk=None
        ):
        """
        Compute DANSE or TI-DANSE network-wide filters.

        Parameters:
        - Pk (list): List of matrices representing the filters for each neighbor node.
        - w (list): List of matrices representing the weights for each neighbor node.
        - algo (str): Algorithm to use ('DANSE', 'TI-DANSE', or 'TI-DANSE+').
        - Tk (list): TI-DANSE+ transformation matrices.
        
        Returns:
        - wNetWide (list): List of matrices representing the network-wide filters for each node.
        """       
        wNetWide = [None for _ in range(self.cfg.K)]
        concatAxis = 1
        for k in range(self.cfg.K):
            t = ()  # tuple: one entry = network-wide filters for one neighbor
            idxNeigh = 0
            for q in range(self.cfg.K):
                if q != k:
                    if algo == 'DANSE':
                        t += (Pk[q] @ w[k][
                            ...,
                            self.cfg.Mk + idxNeigh * self.cfg.Q:\
                                self.cfg.Mk + (idxNeigh + 1) * self.cfg.Q,
                            :
                        ],)
                    elif algo == 'TI-DANSE':
                        t += (Pk[q] @ w[k][
                            ...,
                            self.cfg.Mk:,
                            :
                        ],)
                    elif algo == 'TI-DANSE+':
                        t += (Pk[q] @ np.linalg.pinv(Tk[k]),)
                    elif algo in ['Centr.', 'Local']:
                        return w  # MWF: filters _are_ network-wide filters!
                    else:
                        raise ValueError(f'Unknown algorithm: {algo}')
                    idxNeigh += 1
                else:
                    t += (w[k][..., :self.cfg.Mk, :],)
            wNetWide[k] = np.concatenate(t, axis=concatAxis)
        return wNetWide

    def selection_matrix(self, n: int):
        """Generate selection matrix for dimension `n`."""
        return np.concatenate((  # selection matrix for filters at root node
            np.eye(self.cfg.Q),
            np.zeros((n - self.cfg.Q, self.cfg.Q))
        ), axis=0)

    def compute_tidansep_dim(self, wasn: WASN):
        """Compute number of inputs per node for TI-DANSE+."""
        DkPlus = np.zeros(self.cfg.K, dtype=int)
        for k in range(self.cfg.K):
            wasnPruned = wasn.prune_tree(k)
            nNeighs_k = np.sum(wasnPruned.adjacency, axis=0)[k]
            DkPlus[k] = self.cfg.Mk + self.cfg.Q * nNeighs_k
        return DkPlus
    
    def compute_Cmat(self, k, Pk, wasn: WASN=None, algo='DANSE') -> np.ndarray:
        """Compute the C matrix for DANSE, TI-DANSE, or TI-DANSE+."""
        
        singleTapFlag = Pk[0].ndim != 3
        if singleTapFlag:  # not WOLA domain (single-tap filters time-domain)
            Pk = [p[np.newaxis, ...] for p in Pk]

        if algo == 'DANSE':
            Cmat = self.compute_Cmat_danse(k, Pk)	
        elif algo == 'TI-DANSE':
            Cmat = self.compute_Cmat_tidanse(k, Pk)
        elif algo == 'TI-DANSE+':
            Cmat = self.compute_Cmat_danseplus(k, Pk, wasn)
        
        if singleTapFlag:
            return Cmat[0, ...]  # when working with single-tap, collapse the first dimension
        else:
            return Cmat

    def compute_Cmat_danse(self, k, Pk) -> np.ndarray:
        """
        Compute the C matrix for the DANSE algorithm, as defined in [2], such 
        that $~y = C^H y$.
        """
        nBins = Pk[0].shape[0]
        Ak = np.tile(self.compute_A_mat(k), (nBins, 1, 1))
        Bk_up = np.zeros((nBins, k * self.cfg.Mk, k * self.cfg.Q), dtype=Pk[0].dtype)
        Bk_down = np.zeros((
            nBins,
            (self.cfg.K - k - 1) * self.cfg.Mk,
            (self.cfg.K - k - 1) * self.cfg.Q
        ), dtype=Pk[0].dtype)
        for f in range(nBins):   # for-loop for now
            Bk_up[f, ...] = sla.block_diag(*[
                Pk[q][f, ...]
                for q in range(k)
            ])
            Bk_down[f, ...] = sla.block_diag(*[
                Pk[q][f, ...]
                for q in range(k + 1, self.cfg.K)
            ])
        Bk = np.zeros((
            nBins,
            self.cfg.K * self.cfg.Mk,
            (self.cfg.K - 1) * self.cfg.Q
        ), dtype=Pk[0].dtype)
        Bk[:, :k * self.cfg.Mk, :k * self.cfg.Q] = Bk_up
        Bk[:, (k + 1) * self.cfg.Mk:, k * self.cfg.Q:] = Bk_down
        Ck = np.concatenate((Ak, Bk), axis=2)
        return Ck
    
    def compute_Cmat_tidanse(self, k, Pk):
        """
        Compute the C matrix for the TI-DANSE algorithm, such that $~y = C^H y$.
        """
        nBins = Pk[0].shape[0]
        Ak = np.tile(self.compute_A_mat(k), (nBins, 1, 1))
        Bk = np.concatenate([
            Pk[q] if q != k
            else np.zeros((nBins, self.cfg.Mk, self.cfg.Q), dtype=Pk[0].dtype)
            for q in range(self.cfg.K) 
        ], axis=1)
        Ck = np.concatenate((Ak, Bk), axis=2)
        return Ck
    
    def compute_Cmat_danseplus(self, k, Pk, wasn: WASN):
        """
        Compute the C matrix for the TI-DANSE+ algorithm such that $~y = C^H y$.
        """
        nBins = Pk[0].shape[0]
        Ak = np.tile(self.compute_A_mat(k), (nBins, 1, 1))
        allNodesIdx = np.arange(self.cfg.K)
        neighbors = allNodesIdx[wasn.adjacency[k, :].astype(bool)]
        Bk = np.zeros((
            nBins,
            self.cfg.K * self.cfg.Mk,
            self.cfg.Q * len(neighbors)
        ), dtype=Pk[0].dtype)
        for q in range(len(neighbors)):
            branch = allNodesIdx[wasn.lq == neighbors[q]]
            Bkb = np.concatenate([
                np.zeros((nBins, self.cfg.Mk, self.cfg.Q), dtype=Pk[0].dtype)
                if q == k or q not in branch
                else Pk[q]
                for q in range(self.cfg.K)
            ], axis=1)
            Bk[:, :, q * self.cfg.Q:(q + 1) * self.cfg.Q] = Bkb
        Ck = np.concatenate((Ak, Bk), axis=2)
        return Ck

    def compute_A_mat(self, k):
        Amat = np.concatenate((
            np.zeros((k * self.cfg.Mk, self.cfg.Mk)),
            np.eye(self.cfg.Mk),
            np.zeros(((self.cfg.K - k - 1) * self.cfg.Mk, self.cfg.Mk))
        ), axis=0)
        return Amat
    
    def theoretical_centr_scms_wola(
            self,
            wasn: WASN,
            selfNoise: list
        ):
        """Compute theoretical centralized SCMs for WOLA-domain processing in a 
        given static WASN configuration."""
        vCentr = np.concatenate(tuple(selfNoise), axis=0)
        vCentr_stft = self.cfg.get_stft(vCentr)
        nPosFreq = vCentr_stft.shape[0]

        # RIRs DFT
        rirs_dft = np.array([
            [np.fft.fft(r, n=self.cfg.ndft) for r in rir]
            for rir in wasn.rirs
        ]).transpose(2, 0, 1)  # [F x M x S]
        if self.cfg.singleBin is None:
            rirs_dft = rirs_dft[:nPosFreq, :, :]
        else:
            rirs_dft = rirs_dft[[self.cfg.singleBin], :, :]
        # Latent SCMs for localized sources
        slat_stft = np.array([
            self.cfg.get_stft(s) for s in wasn.latent.globalDesired
        ]).transpose(1, 0, 2)  # [F x S x T]
        nlat_stft = np.array([
            self.cfg.get_stft(n) for n in wasn.latent.globalNoise
        ]).transpose(1, 0, 2)
        if self.cfg.singleBin is not None:
            slat_stft = slat_stft[[self.cfg.singleBin], :, :]
            nlat_stft = nlat_stft[[self.cfg.singleBin], :, :]
        RssLat = np.array([
            np.diag(np.mean(np.abs(slat_stft[f, ...]) ** 2, axis=-1))
            for f in range(slat_stft.shape[0])
        ])
        RnnLat = np.array([
            np.diag(np.mean(np.abs(nlat_stft[f, ...]) ** 2, axis=-1))
            for f in range(nlat_stft.shape[0])
        ])
        # Desired-only centralized SCM
        Rssc = rirs_dft[..., :self.cfg.nGlobDes] @\
            RssLat @\
            rirs_dft[..., :self.cfg.nGlobDes].conj().transpose(0, 2, 1)
        # Localized-noise-only centralized SCM
        Rnncloc = rirs_dft[..., self.cfg.nGlobDes:] @\
            RnnLat @\
            rirs_dft[..., self.cfg.nGlobDes:].conj().transpose(0, 2, 1)
        # Self-noise centralized SCM (latent = actual)
        if self.cfg.singleBin is not None:
            vCentr_stft = vCentr_stft[[self.cfg.singleBin], ...]
        Rvvc = np.array([
            np.diag(np.mean(np.abs(vCentr_stft[f, ...]) ** 2, axis=-1))
            for f in range(vCentr_stft.shape[0])
        ])
        # Noise-only centralized SCM
        Rnnc = Rnncloc + Rvvc
        # Entire (i.e., mic signal) centralized SCM
        Ryyc = Rssc + Rnnc
        return Ryyc, Rnnc, Rssc

    def filtup(
            self,
            Ryy: np.ndarray,
            Rnn: np.ndarray,
            e: np.ndarray,
        ) -> np.ndarray:
        """Filter update function."""
        w = np.linalg.inv(Ryy) @ (Ryy - Rnn)
        if e is not None:
            return w @ e
        else:
            return w

    def get_mse_w(self, w, wc, Q, u=None):
        """
        Computes the MSE of the filter weights with respect to the centralized
        filter weights.

        w = distributed network-wide filter weights
        wc = centralized filter weights
        Q = algorithm dimension (= # of channels in desired signal)
        u = updating node index. If not None, just compute the MSEw for node `u`.
        """
        if u is not None:
            rangeK = [u] * self.cfg.K
        else:
            rangeK = range(self.cfg.K)
        return [
            1 / (self.cfg.Mk * self.cfg.K * Q * w[k].shape[0]) * np.sum(np.array([
                np.linalg.norm(
                    w[k][ii, ...] - wc[k][ii, ...],
                    ord='fro'
                ) ** 2
                for ii in range(w[k].shape[0])
            ])) for k in rangeK
        ]
    
    def build_signals(self, wasn: WASN, rng=np.random.RandomState()):
        """Build the signals for the WASN(s)."""
        # Get microphone signals
        s_g, n_g = wasn.get_wet_signals()
        # Include self-noise
        selfNoise = [None for _ in range(self.cfg.K)]
        leng = int(np.ceil(self.cfg.fs * self.cfg.TmaxOnlineMode))
        for k in range(self.cfg.K):
            # Taking the power of the mixed desired signal contributions
            # in the first sensor of the node as reference.
            mixDesPow = np.mean(np.abs(s_g[k][0, :]) ** 2)
            snTargetPow = self.cfg.selfNoiseFactor * mixDesPow
            snBasis = myrand((self.cfg.Mk, leng), rng, complex=False)
            snBasisPow = np.mean(np.abs(snBasis) ** 2)
            # Adjust the self-noise signal so that its power matches the target power
            selfNoise[k] = np.sqrt(snTargetPow / snBasisPow) * snBasis

        s = [s_g[k] for k in range(self.cfg.K)]  # desired contribution
        n = [n_g[k] + selfNoise[k] for k in range(self.cfg.K)]  # noise contribution

        # Compute SNR for each node
        snr = np.array([
            10 * np.log10(np.mean(np.abs(s[k]) ** 2) / np.mean(np.abs(n[k]) ** 2))
            for k in range(self.cfg.K)
        ])
        print(f'Avg. SNR across the {self.cfg.K} nodes: {np.mean(snr):2f} dB.')
        return s_g, n_g, selfNoise

    def online_up_scm(self, R, y: np.ndarray):
        """
        Update the sample covariance matrix in an online fashion
        using sample data `y`.

        Parameters
        ----------
        R : np.ndarray
            The current sample covariance matrix.
        y : np.ndarray
            The sample data.
        """
        yyH = y @ y.transpose(0, 2, 1).conj() / y.shape[2]
        return self.cfg.beta * R + (1 - self.cfg.beta) * yyH

@dataclass
class Algorithm(Runner):
    wasn: WASN = None # WASN object (required field -- to input on instanciation)
    i: int = 0   # iteration index
    i_frames: list[int] = field(default_factory=list)  # frame indices at which an iteration has occurred
    nRyyUp: list = field(default_factory=list)  # update counter for Ryy
    nRnnUp: list = field(default_factory=list)  # update counter for Rnn
    w: list = field(default_factory=list)  # internal filter weights (used for target signal estimation)
    mse_w: np.ndarray = field(default_factory=lambda: np.array([]))  # single-frame MSEw per node
    Ryy: list = field(default_factory=list)  # Ryy SCM
    Rnn: list = field(default_factory=list)  # Rnn SCM
    Rss: list = field(default_factory=list)  # Rss SCM
    dim: int = 0  # algorithm dimension
    Pk: list = field(default_factory=list)  # fusion matrices for each node
    k: int = 0  # updating node index
    e: np.ndarray = field(default_factory=lambda: np.array([]))  # selection matrix

    def __post_init__(self):
        c = self.cfg
        self.nRyyUp = [np.zeros(c.nPosFreqs, dtype=int) for _ in range(self.cfg.K)]
        self.nRnnUp = [np.zeros(c.nPosFreqs, dtype=int) for _ in range(self.cfg.K)]
        self.mse_w = np.zeros(c.K)
        indivPk = np.zeros((c.Mk, c.Q))
        indivPk[:c.Q, :] = np.eye(c.Q)
        self.Pk = [np.tile(indivPk, (c.nPosFreqs, 1, 1)) for _ in range(c.K)]
        self.i_frames = []

    def init_scms(self, algo=''):
        """Initialize SCMs."""
        c = self.cfg
        def _init_scm(M):
            # return pos_def_hermitian_full_rank((nPosFreq, M, M), rng, complex=makeComplex)
            return np.zeros((c.nPosFreqs, M, M), dtype=complex)
        if algo == 'Centr.':
            # One set of SCMs for all nodes
            self.Ryy = _init_scm(self.dim[0])
            self.Rnn = _init_scm(self.dim[0])
            self.Rss = _init_scm(self.dim[0])
        else:
            # One set of SCMs per node
            self.Ryy = [_init_scm(self.dim[k]) for k in range(c.K)]
            self.Rnn = [_init_scm(self.dim[k]) for k in range(c.K)]
            self.Rss = [_init_scm(self.dim[k]) for k in range(c.K)]

    def update_scm(self, frame: SignalsCollection, scmType='y'):
        """
        Performs one online update of an SCM.

        Parameters
        ----------
        frame : SignalsCollection
            Signals from all nodes at current frame.
        scmType : str
            Type of SCM to update ('y', 's', or 'n').
        """
        c = self.cfg  # alias for convenience

        if scmType == 'y':
            scm = self.Ryy
        elif scmType == 'n':
            scm = self.Rnn
        elif scmType == 's':
            scm = self.Rss
        
        # Perform SCM(s) update
        if self.id == 'Centr.':
            scm = self.online_up_scm(
                scm,
                frame.wd[scmType]
            )
        else:
            for q in range(c.K):
                if self.id == 'TI-DANSE+' and q != self.k:
                    continue  # Only update the "updating node"'s SCM with
                    # TI-DANSE+ (because only the updating node has access to
                    # its full observation vector yTilde, in this algorithm,
                    # unlike in DANSE and TI-DANSE)
                scm[q] = self.online_up_scm(
                    scm[q],
                    frame.get_tilde(scmType)[q]
                )


    def run_single_frame(self, frame: SignalsCollection):
        """
        Runs algorithm for a single frame.

        Parameters
        ----------
        frame : SignalsCollection
            Signals from all nodes.
        """
        self.run_algo_specific(frame)


@dataclass
class MWF(Algorithm):
    id: str = 'Centr.'  # algorithm ID

    def __post_init__(self):
        c = self.cfg
        # Init parent class
        super().__post_init__()
        # Init algorithm-specific attributes
        self.dim = c.K * [c.Mk * c.K]  # algorithm dimension
        self.init_scms(self.id)
        self.w = [myrand((c.nPosFreqs, self.dim[k], c.Q), complex=True) for k in range(c.K)]
    
    def run_algo_specific(self, frame: SignalsCollection,):
        c = self.cfg  
        # Centralized case
        frameCentr = frame.centralize()
        # Use theoretical SCMs (fields of parent class `Run`, computed in
        # `online_run_wola`)
        self.Ryy = copy.deepcopy(self.Ryyct)
        self.Rss = copy.deepcopy(self.Rssct)
        self.Rnn = copy.deepcopy(self.Rnnct)

        # Filter weights computation (usable for all nodes)
        wC = self.filtup(
            self.Ryy,
            self.Ryy - self.Rss if c.useRss else self.Rnn,
            np.eye(c.M),
        )
        for q in range(c.K):
            # Node-specific selection matrix
            ek = np.zeros((c.M, c.Q))
            ek[q * c.Mk:q * c.Mk + c.Q, :] = np.eye(c.Q)
            self.w[q] = wC @ ek


@dataclass
class DANSEcommon(Algorithm):
    id: str = ''  # algorithm ID (placeholder here)

    def up_scms(self, frame: SignalsCollection, algo):
        """SCM update function, common to all DANSE-like algorithms."""
        # Compute fused signals
        frame.fuse(self.Pk)
        # Algorithm-specific observation vectors 
        if algo == 'TI-DANSE+':
            frame.build_obs_vectors(
                algo=algo,
                k=self.k,
                wasnsPruned=self.wasnPruned,
            )
        else:
            frame.build_obs_vectors(algo=algo)
        
        # Algorithm-specific theoretical SCMs
        if algo == 'TI-DANSE+':
            Ck = self.compute_Cmat(
                self.k,
                self.Pk,
                self.wasnPruned[self.k],
                algo=algo
            )
        else:
            Ck = self.compute_Cmat(self.k, self.Pk, algo=algo)
        self.Ryy[self.k] = Ck.transpose(0, 2, 1).conj() @ self.Ryyct @ Ck
        self.Rss[self.k] = Ck.transpose(0, 2, 1).conj() @ self.Rssct @ Ck
        self.Rnn[self.k] = Ck.transpose(0, 2, 1).conj() @ self.Rnnct @ Ck
    
    def run_frame(
            self,
            frame: SignalsCollection,
            algo: str,
        ):
        """Single frame operations common to DANSE and TI-DANSE."""
        if algo not in ['DANSE', 'TI-DANSE']:
            raise ValueError('`algo` should be "DANSE" or "TI-DANSE".')
        
        c = self.cfg
        # Update SCMs based on `frame`
        self.up_scms(frame, algo)

        # Update filter
        self.w[self.k] = self.filtup(
            self.Ryy[self.k],
            self.Ryy[self.k] - self.Rss[self.k] if c.useRss else self.Rnn[self.k],
            self.e,
        )

        if algo == 'DANSE':
            # Fusion matrix for DANSE: Wkk
            self.Pk[self.k] = self.w[self.k][:, :c.Mk, :]
        elif algo == 'TI-DANSE':
            # Fusion matrix for TI-DANSE: Wkk.Gk^{-1}
            self.Pk[self.k] = self.w[self.k][:, :c.Mk, :] @\
                    np.linalg.pinv(self.w[self.k][:, c.Mk:, :])
        self.k = (self.k + 1) % c.K
        self.i += 1

@dataclass
class DANSE(DANSEcommon, Algorithm):
    id: str = 'DANSE'  # algorithm ID

    def __post_init__(self):
        c = self.cfg
        # Init parent class
        super().__post_init__()
        # Init algorithm-specific attributes
        self.dim = c.K * [c.Dk]  # algorithm dimension
        self.init_scms()
        self.e = self.selection_matrix(self.dim[0])
        self.w = [myrand((c.nPosFreqs, self.dim[k], c.Q), complex=True) for k in range(c.K)]
        
    def run_algo_specific(self, frame: SignalsCollection):
        self.run_frame(frame, algo=self.id)


@dataclass
class TIDANSE(DANSEcommon, Algorithm):
    id: str = 'TI-DANSE'  # algorithm ID

    def __post_init__(self):
        c = self.cfg
        # Init parent class
        super().__post_init__()
        # Init algorithm-specific attributes
        self.dim = c.K * [c.DkTI]  # algorithm dimension
        self.init_scms()
        self.e = self.selection_matrix(self.dim[0])
        self.w = [myrand((c.nPosFreqs, self.dim[k], c.Q), complex=True) for k in range(c.K)]
        
    def run_algo_specific(self, frame: SignalsCollection):
        self.run_frame(frame, algo=self.id)


@dataclass
class TIDANSEplus(DANSEcommon, Algorithm):
    id: str = 'TI-DANSE+'  # algorithm ID
    wasnPruned: dict[int, WASN] = field(default_factory=dict)  # Pruned WASN objects
    Tk: list = field(default_factory=list)  # transformation matrices

    def __post_init__(self):
        if self.wasn is None:
            raise ValueError('WASN object required for TI-DANSE+ algorithm.')
        c = self.cfg
        # Init parent class
        super().__post_init__()
        # Init algorithm-specific attributes
        self.dim = self.compute_tidansep_dim(self.wasn)  # algorithm dimension
        self.init_scms(self.id)
        self.Tk = [np.tile(np.eye(c.Q), (c.nPosFreqs, 1, 1)) for _ in range(c.K)]
        self.w = [myrand((c.nPosFreqs, self.dim[k], c.Q), complex=True) for k in range(c.K)]
        self.wasnPruned = dict([
            (q, self.wasn.prune_tree(q))
            for q in range(c.K)
        ])

    def run_algo_specific(self, frame: SignalsCollection):
        c = self.cfg
        # Select relevant WASN 
        wasnPrunedCurr_k = self.wasnPruned[self.k]
        # Set of children of root node `k`
        lSet_k_p = np.where(wasnPrunedCurr_k.adjacency[self.k, :] == 1)[0]
        # Update SCMs based on `frame` data
        self.up_scms(frame, self.id)

        # Update TI-DANSE+ filter at root node
        eTIplusRoot = self.selection_matrix(self.Ryy[self.k].shape[1])
        self.w[self.k] = self.filtup(
            self.Ryy[self.k],
            self.Ryy[self.k] - self.Rss[self.k] if c.useRss else self.Rnn[self.k],
            eTIplusRoot,
        )
        # Define H-matrices
        Hmat = [None for _ in range(c.K)]
        for q in range(c.K):
            if q == self.k:
                pass  # no need for H-matrix at root node
                # Hmat[q] = np.tile(np.eye(c.Q), (c.nPosFreqs, 1, 1))
            else:
                lqCurr = wasnPrunedCurr_k.lq[q]
                idxLq = np.where(lSet_k_p == lqCurr)[0][0]  # index of `lqCurr` in set of root children
                Hmat[q] = self.w[self.k][
                    :, c.Mk + idxLq * c.Q: c.Mk + (idxLq + 1) * c.Q, :
                ]
        
        # Fusion matrices for TI-DANSE+
        for q in range(c.K):
            Iq = np.tile(np.eye(c.Q), (c.nPosFreqs, 1, 1))
            self.Tk[q] = Iq if q == self.k else self.Tk[q] @ Hmat[q] # sequential filter update
            self.Pk[q] = self.w[q][:, :c.Mk, :] @ self.Tk[q]
        
        self.k = (self.k + 1) % c.K
        self.i += 1

@dataclass
class LatentSignals:
    globalDesired: list = field(default_factory=list)
    globalNoise: list = field(default_factory=list)
    localDesired: list = field(default_factory=list)
    localNoise: list = field(default_factory=list)
    cfg: Parameters = field(default_factory=lambda: Parameters())

    def __post_init__(self):
        self.get_latent_signals()

    def get_latent_signals(self):
        """
        Returns the latent source signals for each desired and noise source.
        Depending on the configuration, the latent signals are either generated
        randomly or read from a file.
        """
        rng = np.random.RandomState(self.cfg.seed)
        # Determine number of samples
        n = int(np.ceil(self.cfg.fs * self.cfg.TmaxOnlineMode))
        # Generate or read latent signals
        for s in range(self.cfg.nGlobDes):
            self.globalDesired.append(
                self.get_latent_sig(n, rng=rng)
            )
        for s in range(self.cfg.nGlobNoi):
            self.globalNoise.append(
                self.get_latent_sig(n, rng=rng)
            )

    def get_latent_sig(self, n, rng=np.random.RandomState()):
        """
        Generates or reads latent signal from file.
        --n [int] - number of samples
        --rng [np.random.RandomState] - random number generator
        Returns
        --signal [np.array] - latent signal
        """
        return myrand((n,), rng=rng, complex=False)
        
def inner(w: np.ndarray, y: np.ndarray):
    """
    Inner product operation.
    
    w is a transformation matrix of shape [(F x )Mk x Q].
    y is a signal STFT of shape [(F x )Mk x T].
    Output is of shape [(F x )Q x T].
    """
    if w.ndim == 2:
        return w.T @ y
    elif w.ndim == 3:
        return w.transpose(0, 2, 1) @ y

