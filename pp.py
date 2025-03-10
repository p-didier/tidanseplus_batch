# Purpose of script:
# This script post-processes data and plots the results for the EUSIPCO 2025
# submission: "Improved Distributed Adaptive Node-Specific Signal Estimation for
# Topology-Unconstrained Wireless Acoustic Sensor Networks" by Paul Didier,
# Toon van Waterschoot, Simon Doclo, Jörg Bitzer, and Marc Moonen.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
import package.mypalettes as pal
from package.utils import Parameters

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

EXPORT_FIGURES = True  # if True, export figures to PDF and PNG

MAXITER = 200  # maximum number of iterations

PALETTE = 'seabed'  # color palette (as defined in mypalettes.py)

def main():
    """Main function (called by default when running script)."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the post-processing script for a particular folder.")
    parser.add_argument('--path', type=str, required=True, help="Path to the simulation results")
    args = parser.parse_args()
    folderPath = args.path

    def _export(f: plt.Figure, folder: str, name: str, skipPdf: bool=True):
        if EXPORT_FIGURES and f is not None:
            f.savefig(f'{folder}/{name}.png', dpi=300)
            f.savefig(f'{folder}/{name}.svg', dpi=300)
            if not skipPdf:
                f.savefig(f'{folder}/{name}.pdf')
            print(f'Figure "{name}" exported to {folder}.')

    if Path(f'{folderPath}/data.pkl').is_file():
        # Load pre-processed data
        print(f'Loading pre-processed data from {folderPath}...')
        with open(f'{folderPath}/data.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        # List subfolders
        subfolders = [p for p in Path(folderPath).iterdir() if p.is_dir()]
        if len(subfolders) == 0:
            print('SKIPPING: No subfolders found.')
            return 1
        
        # Compile data
        Cvals, MCvals = parse_subfolders(folderPath)
        data = dict([(c, dict([(mc, None) for mc in MCvals])) for c in Cvals])
        for ii, sf in enumerate(subfolders):
            print(f'Processing subfolder {ii + 1}/{len(subfolders)}...')
            dataFile = f'{sf}/data.pkl'
            if not Path(dataFile).exists():
                print(f'SKIPPING: Data file "{dataFile}" not found.')
                continue
            c, mc = parse_subfolder_name(sf.name)
            with open(dataFile, 'rb') as f:
                tmp = pickle.load(f)
                data[c][mc] = {
                    'cfg': tmp['cfg'],
                    'mse_w': tmp['mse_w'],
                }

        # Export `data` to a pickle file
        with open(f'{folderPath}/data.pkl', 'wb') as f:
            pickle.dump(data, f)

    # MSE_W
    fig = msew_plot(data)
    _export(fig, folderPath, 'mse_w')

    return 0

def msew_plot(data: dict):
    """
    MSE_W cost curves as function of frame index.
    
    Parameters
    ----------
    data : dict
        Dictionary containing the data.
        > Keys are the values of C.
        > Values are dictionaries with keys being the values of MC and values
            being the data dictionaries.
    """
    # Hard-coded things
    algsOfInterest = {
        'DANSE': 'k',
        'TI-DANSE': 'k',
        'TI-DANSE+': 'palette',
    }
    palette = pal.get_palette(PALETTE)

    def process_color(color: str, idx: int):
        if color == 'palette':
            return palette[idx]
        else:
            return color
    
    # Geometrical mean
    mymean = lambda x: np.exp(np.mean(np.log(x), axis=0))

    Cvals = list(data.keys())
    MCvals = list(data[Cvals[0]].keys())

    c: Parameters = data[Cvals[0]][MCvals[0]]['cfg']

    # Process: mean MSE_W across MCs and nodes
    dataToPlot = dict([(alg, None) for alg in algsOfInterest.keys()])
    for alg in algsOfInterest.keys():
        tmp = [
            mymean(np.array([
                np.mean(data[c][mc]['mse_w'][alg], axis=1)  # mean across nodes
                for mc in MCvals
            ]))  # mean across MCs
            for c in Cvals
        ]
        if len(tmp[0]) > MAXITER:
            tmp = [t[:MAXITER] for t in tmp]
        dataToPlot[alg] = tmp

    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'x', '+', '*', '|', '_']

    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(6.5, 3)
    markerCounter = 0
    for algName in algsOfInterest.keys():
        for ii in range(len(Cvals)):
            lw = 2  # linewidth
            if algName == 'TI-DANSE+':
                lab = algName + f' (C={p_to_dot(Cvals[ii])})'
                lw = 1.2
            elif ii > 0:
                continue
            else:
                lab = algName
            col = process_color(algsOfInterest[algName], ii)
            axes.semilogy(
                dataToPlot[algName][ii],
                color=col,
                marker=markers[markerCounter],
                label=lab,
                linewidth=lw,
                markersize=4,
                markevery=25,
                markerfacecolor=col,
            )
            markerCounter += 1
    axes.legend()
    axes.set_xlabel('Frame index $l$')
    axes.set_ylabel(f'MSE$_W^i$')
    fig.tight_layout()
    plt.show(block=False)

    return fig

def parse_subfolder_name(subfolder: str):
    """Parse subfolder name and return C and MC values."""
    Cval = float(p_to_dot(subfolder.split('_')[0][1:]))
    MCval = int(subfolder.split('_')[1][2:])
    return Cval, MCval


def parse_subfolders(folder: str):
    """Parse subfolders and return list of subfolders."""
    subfolders = [p.name for p in Path(folder).iterdir() if p.is_dir()]
    if len(subfolders) == 0:
        print('SKIPPING: No subfolders found.')
        return None
    # Parse subfolder names
    Cvals, MCvals = [], []
    for sf in subfolders:
        Cval, MCval = parse_subfolder_name(sf)
        Cvals.append(Cval)
        MCvals.append(MCval)
    # Sort and make unique
    Cvals = sorted(list(set(Cvals)))
    MCvals = sorted(list(set(MCvals)))
    return Cvals, MCvals



def p_to_dot(p: float):
    """Convert float to string with dot."""
    return str(p).replace('p', '.')


if __name__ == '__main__':
    sys.exit(main())