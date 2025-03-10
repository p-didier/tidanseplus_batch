from .utils import *
import pyroomacoustics as pra

@dataclass
class AcousticScenario:
    """Acoustic scenario class."""
    cfg: Parameters

    def setup_constellation(self, minNodeDist=0, rng=np.random.RandomState()):
        """Set up the constellation of nodes."""
        c = self.cfg  # alias for convenience
        nodePositions = []
        counter = 0
        while len(nodePositions) < c.K:
            currPos = rng.uniform(c.mindWalls, c.rd - c.mindWalls, (1, c.dim))
            if len(nodePositions) == 0:
                nodePositions.append(currPos)
            else:
                dists = np.linalg.norm(nodePositions - currPos, axis=1)
                if np.all(dists > minNodeDist):
                    nodePositions.append(currPos)
            counter += 1
            if counter > 1000:
                raise ValueError('Could not generate the desired number of nodes. Try increasing the room dimensions or decreasing the minimum inter-node distance.')
        nodePositions = np.concatenate(nodePositions, axis=0)

        # Check that the nodes are inside the room
        if np.any(nodePositions < self.cfg.mindWalls) or np.any(nodePositions > self.cfg.rd - self.cfg.mindWalls):
            raise ValueError('The nodes are not inside the room. Try increasing the room dimensions or decreasing the minimum distance to the walls.')
        return nodePositions
    
    def set_target_connectivity(self, aMat, targetConn=None, rng=np.random.RandomState()):
        """Ensure target degree of connectivity in non-fully connected network."""
        n1s_fc = self.cfg.K * (self.cfg.K - 1)  # number of 1's in adjacency matrix for full connectivity
        n1s_mc = 2 * self.cfg.K  # number of 1's in adjacency matrix for minimum connectivity
        currConn = compute_connectivity(self.cfg, aMat)
        flagPrint = True
        if targetConn is not None:
            rngConn = np.random.RandomState(rng.randint(0, 1000))  # for reproducibility
            while currConn > targetConn and currConn - 2 / (n1s_fc - n1s_mc) >= targetConn:
                if flagPrint:
                    flagPrint = False
                # Remove two connections to make the graph less connected
                # (ensuring that the graph remains connected)
                k1, k2 = rngConn.choice(self.cfg.K, 2, replace=False)
                if aMat[k1, k2] == 1:
                    aMat[k1, k2] = 0
                    aMat[k2, k1] = 0
                # Check if the graph is still connected
                while not np.linalg.matrix_power(aMat, self.cfg.K).all():
                    aMat[k1, k2] = 1
                    aMat[k2, k1] = 1
                    k1, k2 = rngConn.choice(self.cfg.K, 2, replace=False)
                    if aMat[k1, k2] == 1:
                        aMat[k1, k2] = 0
                        aMat[k2, k1] = 0
                currConn = compute_connectivity(self.cfg, aMat)
            while currConn < targetConn:
                if flagPrint:
                    flagPrint = False
                # Add two connections to make the graph more connected
                k1, k2 = rngConn.choice(self.cfg.K, 2, replace=False)
                if aMat[k1, k2] == 0:
                    aMat[k1, k2] = 1
                    aMat[k2, k1] = 1
                currConn = compute_connectivity(self.cfg, aMat)
        return aMat
    
    def generate_sensors(self, nodePositions, rng=np.random.RandomState()):
        """Generate sensor positions."""
        sensorPositions = np.zeros((self.cfg.K, self.cfg.Mk, self.cfg.dim))
        for k in range(self.cfg.K):
            while True:
                pos = nodePositions[k, :] + rng.uniform(
                    -self.cfg.d / 2, self.cfg.d / 2, (self.cfg.Mk, self.cfg.dim)
                )
                # Check if the sensors are inside the room
                if np.all(pos >= self.cfg.mindWalls) and np.all(pos <= self.cfg.rd - self.cfg.mindWalls):
                    break
            sensorPositions[k, :, :] = pos
        return sensorPositions

    def generate_sources(self, sensorPositions: np.ndarray, rng=np.random.RandomState()):
        """Generate source positions."""
        c = self.cfg  # alias for convenience
        rngConn = np.random.RandomState(rng.randint(0, 1000))  # for reproducibility
        nSources = c.nGlobDes + c.nGlobNoi
        sources = np.zeros((nSources, c.dim))
        # Precompute the sensor positions as a single array for faster distance calculations
        sensorPositions_flat = sensorPositions.reshape(-1, c.dim)
        for s in range(nSources):
            counter = 0
            while True:
                # Generate a candidate position
                sPosCurr = rngConn.uniform(c.mindWalls, c.rd - c.mindWalls, c.dim)
                # Calculate distances to all sensors
                distances = np.linalg.norm(sensorPositions_flat - sPosCurr, axis=1)
                # Check if the candidate position is valid
                if np.all(distances >= c.mindSourceSensor):
                    sources[s, :] = sPosCurr
                    break
                if counter > 1000:
                    raise ValueError('Could not generate the desired number of sources. Try increasing the room dimensions or decreasing the minimum distance between sources and sensors.')
        
        pos = {
            'gd': sources[:c.nGlobDes, :],
            'gn': sources[c.nGlobDes:c.nGlobDes + c.nGlobNoi, :]
        }
        return pos

    def gen_steering_mats(self, sensorPositions, sourcesPositions: dict[str, np.ndarray]):
        """
        Generate steering matrices based on sensor and source positions.

        Parameters:
        - sensorPositions: sensor positions
        - sourcesPositions: source positions, a dictionary with keys 'gd', 'gn'
        """
        nSources = {
            'gd': self.cfg.nGlobDes,
            'gn': self.cfg.nGlobNoi,
        }
        sm = dict([
            (key, [
                np.zeros((self.cfg.Mk, nSources[key]))
                for _ in range(self.cfg.K)
            ])
            for key in sourcesPositions.keys()
        ])
        for k in range(self.cfg.K):
            for m in range(self.cfg.Mk):
                for s in range(self.cfg.nGlobDes):
                    d = np.linalg.norm(sensorPositions[k, m, :] - sourcesPositions['gd'][s, :])
                    sm['gd'][k][m, s] = 1 / d
                for s in range(self.cfg.nGlobNoi):
                    d = np.linalg.norm(sensorPositions[k, m, :] - sourcesPositions['gn'][s, :])
                    sm['gn'][k][m, s] = 1 / d
        return sm
    
    def generate_wasn(
            self,
            plotit=False,
            targetConn=None,
            minNodeDist=0,  # only used for random constellations
            rng=np.random.RandomState()  # random number generator
        ):
        """
        Generates a random acoustic scenario with a WASN in it.

        Parameters:
        - plotit: whether to plot the WASN
        - targetConn: target connectivity level
        - minNodeDist: minimum distance between nodes
        - rng: random number generator

        Returns:
        - WASN object or list of WASN objects if moving environment.
        """
        c = self.cfg  # alias for convenience

        # Invert Sabine's formula to obtain the parameters for the ISM simulator
        if c.t60 == 0:
            maxOrd = 0
            eAbs = 0.5  # <-- arbitrary
        else:
            eAbs, maxOrd = pra.inverse_sabine(
                c.t60,
                [c.rd, c.rd] if c.dim == 2 else [c.rd, c.rd, c.rd]
            )

        # Static environment
        room = pra.ShoeBox(
            p=[c.rd, c.rd] if c.dim == 2 else [c.rd, c.rd, c.rd],
            fs=c.fs,
            max_order=maxOrd,
            air_absorption=False,
            materials=pra.Material(eAbs),
        )
        
        # Set up the constellation of nodes
        nodesPos = self.setup_constellation(minNodeDist, rng)

        # Non-fully connected network
        aMat, commDist = get_adjacency_matrix(c, nodesPos)
        # Ensure target degree of connectivity
        aMat = self.set_target_connectivity(aMat, targetConn, rng)

        # Generate sensor positions
        sensorsPos = self.generate_sensors(nodesPos, rng)
        
        # Generate source positions
        sourcesPos = self.generate_sources(sensorsPos, rng)

        # Generate latent signals
        latSigs = LatentSignals(cfg=c)
        # Place all microphones in the room at once
        mic_array = pra.MicrophoneArray(sensorsPos.reshape(-1, sensorsPos.shape[-1]).T, room.fs)
        room.add_microphone_array(mic_array)
        # Add sources to the room
        for s in range(c.nGlobDes):
            room.add_source(sourcesPos['gd'][s, :])
        for s in range(c.nGlobNoi):
            room.add_source(sourcesPos['gn'][s, :])
        # Compute the RIRs
        print('Computing RIRs...')
        room.compute_rir()
        print('Done.')
        rirs = room.rir
        
        # Create WASN object
        wasn = WASN(
            nodePositions=nodesPos,
            sensorPositions=sensorsPos,
            globalDesiredPositions=sourcesPos['gd'],
            globalNoisePositions=sourcesPos['gn'],
            adjacency=aMat,
            commDist=commDist,
            rirs=rirs,
            latent=latSigs,
            cfg=c
        )

        if plotit:
            wasn.plot()

        return wasn


def get_adjacency_matrix(cfg: Parameters, nodePositions):
    """Create connection matrix (adjacency matrix) based on node positions."""
    dists = np.zeros((cfg.K, cfg.K))
    for k in range(cfg.K):
        for q in range(cfg.K):
            dists[k, q] = np.linalg.norm(nodePositions[k, :] - nodePositions[q, :])
    # Make binary
    cd = copy.deepcopy(cfg.commDist)
    aMat = np.where(dists < cd, 1, 0)
    # Check if the graph is connected
    while not np.linalg.matrix_power(aMat, cfg.K).all():
        cd += 0.1
        aMat = np.where(dists < cd, 1, 0)
    # Diagonal should be zero
    np.fill_diagonal(aMat, 0)

    return aMat, cd


def compute_connectivity(cfg: Parameters, aMat):
    """Compute connectivity as defined for DANSE+.""" 
    n1s = np.sum(aMat)
    n1s_fc = cfg.K * (cfg.K - 1)  # number of 1's in adjacency matrix for full connectivity
    n1s_mc = 2 * cfg.K  # number of 1's in adjacency matrix for minimum connectivity
    return (n1s - n1s_mc) / (n1s_fc - n1s_mc)
