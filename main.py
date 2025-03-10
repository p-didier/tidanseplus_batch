# This script allows to reproduce Fig. 3 in [1]. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
#
# References:
# [1] Didier, P., van Waterschoot, T., Doclo, S., Bitzer, J., and Moonen, M.,
# Improved Distributed Adaptive Node-Specific Signal Estimation for
# Topology-Unconstrained Wireless Acoustic Sensor Networks. Submitted to
# EUSIPCO 2025.

import sys
import time
import yaml
import pickle
import argparse
import numpy as np
from pathlib import Path
from package.asc import *
from package.online import Run

def main():
    """Main function (called by default when running script)."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the script with a specific configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, 'r') as file:
        cfgDict = yaml.safe_load(file)
    CFG: Parameters = Parameters().load_from_dict(cfgDict)

    if CFG.exportFolder is None:
        CFG.exportFolder = f'{Path.cwd()}/out/at_{time.strftime("%Y%m%d_%H%M%S")}{CFG.exportFolderSuffix}'

    # Save RNG state
    np.random.seed(CFG.seed)
    rngState = np.random.get_state()

    t0 = time.time()
    for idxConn, conn in enumerate(CFG.connectivity):
        # Restore RNG state
        np.random.set_state(rngState)
        for idxMC in range(CFG.nMCbatch):
            print(f'Connectivity {conn} ({idxConn + 1}/{len(CFG.connectivity)}) -- Online run {idxMC + 1}/{CFG.nMCbatch}...')
            out = Run(cfg=CFG).go(
                targetConn=conn,
                rngSeed=CFG.seed + idxMC   # change RNG based on MC index
            )
            foldername = f'{CFG.exportFolder}/C{dot_to_p(np.round(conn, 2))}_MC{idxMC + 1}'
            if not Path(foldername).exists():
                Path(foldername).mkdir(parents=True)
            pickle.dump(out, open(f'{foldername}/data.pkl', 'wb'))
            CFG.dump_to_txt(f'{foldername}/cfg.txt')

    print(f'All done. Elapsed time: {time.time() - t0:.2f} s.')
    print(f'Data exported to "{CFG.exportFolder}".')

    return 0


if __name__ == '__main__':
    sys.exit(main())