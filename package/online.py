from .asc import *

@dataclass
class Run(Runner):
    cfg: Parameters

    def go(self, targetConn, rngSeed=None):
        """Run the DANSE, TI-DANSE, and TI-DANSE+ algorithms
        in the STFT (WOLA) domain."""

        c = self.cfg  # alias for convenience
        
        # Initialize RNG
        if rngSeed is None:
            rngSeed = c.seed
        rng = np.random.RandomState(rngSeed)

        # Generate acoustic scenario and WASN
        asc = AcousticScenario(cfg=c)
        wasn = asc.generate_wasn(targetConn=targetConn, rng=rng)
        
        # Generate signals
        s_g, n_g, selfNoise = self.build_signals(wasn, rng)
        micSignals = SignalsCollection(sigs=[Signal(
            td={
                's': s_g[k],
                'n': n_g[k] + selfNoise[k],
                'y': s_g[k] + n_g[k] + selfNoise[k]
            }
        ) for k in range(c.K)])
        micSignals.compute_stfts(c)  # compute STFTs

        # Instanciate algorithms to be run
        algInstances = dict()
        for algClass in Algorithm.__subclasses__():
            if algClass.id in c.algNames:
                algInstances[algClass.id] = algClass(cfg=c, wasn=wasn)

        # Compute theoretical centralized SCMs, for use in all algorithms
        Ryyct, Rnnct, Rssct = self.theoretical_centr_scms_wola(
            wasn, selfNoise
        )
        for algName in c.algNames:
            algInstances[algName].Ryyct = Ryyct
            algInstances[algName].Rnnct = Rnnct
            algInstances[algName].Rssct = Rssct
        # Useful variables
        nFrames = micSignals.sigs[0].wd['s'].shape[2]  # number of frames        
        # Other array/dict initializations
        mse_w = dict([
            (algName, np.zeros((nFrames + 1, c.K)))
            for algName in c.algNames
        ])

        # >>>>>> MAIN LOOP <<<<<<
        t0 = time.time()  # start time
        for l in range(nFrames):
            print_progress(l, nFrames, algInstances)
            # Current frame signals
            currFrame = micSignals.get_frame(l)
            
            # Run algorithms for this frame
            for algName in c.algNames:
                algInstances[algName].run_single_frame(currFrame)
    
            # Compute MSEw's for this frame, for each algorithm
            for algName in c.algNames:
                if 'Centr.' in c.algNames:
                    if algName not in ['Centr.', 'Local']:
                        wNW = self.get_nw_filters_indiv_algo(
                            Pk=algInstances[algName].Pk,
                            w=algInstances[algName].w,
                            algo=algName,
                            Tk=algInstances[algName].Tk if\
                                algName == 'TI-DANSE+' else None
                        )  # network-wide filters
                        algInstances[algName].mse_w = self.get_mse_w(
                            wNW, algInstances['Centr.'].w, c.Q
                        )
                    
                    mse_w[algName][l, :] = copy.deepcopy(
                        algInstances[algName].mse_w
                    )

        print(f'\nElapsed time: {time.time() - t0}.')
        # >>>> END MAIN LOOP <<<<
        
        return {'mse_w': mse_w,'cfg': c}


def print_progress(l, nFrames, algInstances):
    idList = ''
    for key in algInstances.keys():
        if key not in ['Centr.', 'Local']:
            idList += f'i_{key}={algInstances[key].i} | '
    idList = idList[:-3]
    print(f'[Frame {l + 1}/{nFrames}] {idList}', end='\r')
