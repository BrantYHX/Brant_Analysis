import napari
import tifffile
import numpy as np
from bg_atlasapi import BrainGlobeAtlas

# ── path ──────────────────────────────────────────────────
REG_ATLAS  = r"Y:\public\projects\SaEl_20220201_VIP\2pdata\LC\dreadds\LC_Dreadds_1\brainsaw\registration\reg_01__2026_02_04_a\registered_atlas.tiff"
BOUNDARIES = r"Y:\public\projects\SaEl_20220201_VIP\2pdata\LC\dreadds\LC_Dreadds_1\brainsaw\registration\reg_01__2026_02_04_a\boundaries.tiff"
HISTOLOGY  = r"Y:\public\projects\SaEl_20220201_VIP\2pdata\LC\dreadds\LC_Dreadds_1\brainsaw\downsampled_stacks\025_micron\ds_sample_348dreadds_260202_111849_25_25_ch03_chan_3_green.tif"
BACKGROUND = r"Y:\public\projects\SaEl_20220201_VIP\2pdata\LC\dreadds\LC_Dreadds_1\brainsaw\downsampled_stacks\025_micron\ds_sample_348dreadds_260202_111849_25_25_ch04_chan_4_blue.tif"

# ── load ──────────────────────────────────────────────
print("Loading atlas...")
atlas = BrainGlobeAtlas("allen_mouse_25um")

print("Loading volumes...")
atlas_vol  = tifffile.imread(REG_ATLAS)
boundaries = tifffile.imread(BOUNDARIES)
histology  = tifffile.imread(HISTOLOGY)
background  = tifffile.imread(BACKGROUND)

# ── transform boundaries and atlas_vol ──
atlas_vol  = atlas_vol[::-1, :, ::-1]   # reverse slices + horizontally transform
boundaries = boundaries[::-1, :, ::-1]  # reverse slices + horizontally transform

# ── open napari ───────────────────────────────────────────
viewer = napari.Viewer(title="Brain Region Explorer")

viewer.add_image(histology, name="histology",
                 colormap="green",
                 opacity=1.0,
                 blending="additive")

viewer.add_image(background, name="background",
                 colormap="blue",
                 opacity=1.0,
                 blending="additive")

viewer.add_image(boundaries, name="boundaries",
                 colormap="gray", opacity=0.2,
                 blending="additive")

# ── click & print brain region ──────────────────────────────────
@viewer.mouse_drag_callbacks.append
def identify_region(viewer, event):
    if event.type != 'mouse_press':
        return

    coords = viewer.cursor.position
    z, y, x = int(round(coords[0])), int(round(coords[1])), int(round(coords[2]))

    if not (0 <= z < atlas_vol.shape[0] and
            0 <= y < atlas_vol.shape[1] and
            0 <= x < atlas_vol.shape[2]):
        return

    structure_id = int(atlas_vol[z, y, x])

    if structure_id == 0:
        print(f"  ({z}, {y}, {x})  ->  background / outside atlas")
    else:
        try:
            name    = atlas.structures[structure_id]['name']
            acronym = atlas.structures[structure_id]['acronym']
            print(f"  ({z}, {y}, {x})  ->  {name}  [{acronym}]  (id={structure_id})")
        except KeyError:
            print(f"  ({z}, {y}, {x})  ->  unknown id={structure_id}")

print("\n✅ Ready!")
napari.run()
