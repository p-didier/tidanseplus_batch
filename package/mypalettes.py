import matplotlib.pyplot as plt

PALETTES = {
    # https://coolors.co/d9ed92-b5e48c-99d98c-76c893-52b69a-34a0a4-168aad-1a759f-1e6091-184e77
    'seabed': ["d9ed92","b5e48c","99d98c","76c893","52b69a","34a0a4","168aad","1a759f","1e6091","184e77"],
    # https://coolors.co/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226
    'cool1': ["001219","005f73","0a9396","94d2bd","e9d8a6","ee9b00","ca6702","bb3e03","ae2012","9b2226"],
    # https://coolors.co/f94144-f3722c-f8961e-f9844a-f9c74f-90be6d-43aa8b-4d908e-577590-277da1
    'pastel1': ["f94144","f3722c","f8961e","f9844a","f9c74f","90be6d","43aa8b","4d908e","577590","277da1"]
}
for k, v in PALETTES.items():
    PALETTES[k] = ['#'+c for c in v if c[0] != '#']

# Function to set the custom palette
def set_my_palette(ref: str):
    if ref not in PALETTES:
        raise ValueError(f"Unknown palette reference: {ref}. Possible values:\n{list(PALETTES.keys())}")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTES[ref])

def get_palette(ref: str):
    return PALETTES[ref] if ref in PALETTES else None