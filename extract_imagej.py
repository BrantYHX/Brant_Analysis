import os
import pathlib
import numpy as np
import torch
from suite2p import extraction, detection
from suite2p.io import BinaryFile
from roifile import ImagejRoi
from zipfile import ZipFile
import pandas as pd

roi_type='manrois' 
def load_rois_from_imagej(folder):
    rois = []
    for file in os.listdir(folder):
        if file.endswith('.roi'):
            roi = ImagejRoi.fromfile(plane_path/roi_type/file)
            rois.append(roi)
    return rois

data_dir = r'Y:\\public\\projects\SaEl_20220201_VIP\\2pdata\subj_record.xlsx'
sheet_name = 'LC_DREADDs'

# Define suite2p path
table = pd.read_excel(data_dir, sheet_name=sheet_name)
for ani in [0]:
    for n in range(3): # nplanes
        
        # path_to_suite2p = pathlib.Path("Y:\public\projects\SaEl_20220201_VIP\\2pdata\LC\\vipsilencing\LCR_Vip-\C\LCR_vip-_C\suite2p")
        path_to_suite2p = pathlib.Path(table.iloc[ani][5])
        
        # path_to_suite2p = pathlib.Path('Y:\public\projects\SaEl_20220201_VIP\\2pdata\LC\\vipsilencing\LC12_Vip-\C\suite2p')
        plane = 'plane' + str(n)
        print(path_to_suite2p)
        print(plane)
        plane_path = path_to_suite2p / plane
        path_to_roizip_folder = plane_path/'rois.zip'
        with ZipFile(path_to_roizip_folder,'r') as zObject:
            zObject.extractall(path=plane_path/roi_type)
        zObject.close()
        ops= np.load(os.path.join(plane_path,'ops.npy'),allow_pickle=True).item()
        ops['reg_file'] = os.path.join(plane_path,'data.bin')
        ops['ops_path'] = os.path.join(plane_path,'ops.npy')
        ops['save_path'] = plane_path
        rois=load_rois_from_imagej(plane_path/roi_type)
        stat_manual=[]
        # stat_all=np.load(os.path.join(plane_path,'stat.npy'),allow_pickle=True)
        for roi in rois:
            roi=roi.coordinates()
            xpix=roi[:,0]
            ypix=roi[:,1]
            lam=np.ones(ypix.shape, np.float16) # initiate array of weights for each pixel 
            lam /= lam.sum() #normalize weights
            stat_manual.append({'ypix':ypix.astype(int), 'xpix':xpix.astype(int) ,'lam':lam ,'npix': ypix.shape})# put everything into dictionary
       
        # stat_manual=detection.stats.roi_stats(stat_manual,ops['Ly'],ops['Lx'],aspect=ops.get('aspect', None), diameter=ops['diameter'])
        # Clip BEFORE roi_stats
        for stat in stat_manual:
            stat['ypix'] = np.clip(stat['ypix'], 0, ops['Ly'] - 1).astype(int)
            stat['xpix'] = np.clip(stat['xpix'], 0, ops['Lx'] - 1).astype(int)

        stat_manual = detection.stats.roi_stats(np.array(stat_manual), ops['Ly'], ops['Lx'])

        # Clip AFTER roi_stats (in case it modifies pixel arrays)
        for stat in stat_manual:
            stat['ypix'] = np.clip(stat['ypix'], 0, ops['Ly'] - 1).astype(int)
            stat['xpix'] = np.clip(stat['xpix'], 0, ops['Lx'] - 1).astype(int)

        manual_cell_masks= [
            extraction.masks.create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stat_manual
            ]


        cell_pix = extraction.masks.create_cell_pix(stat_manual,Ly=ops['Ly'], Lx=ops['Lx'])
        manual_neuropil_masks=extraction.masks.create_neuropil_masks(
            ypixs=[stat['ypix'] for stat in stat_manual],
            xpixs=[stat['xpix'] for stat in stat_manual],
            cell_pix=cell_pix,
            inner_neuropil_radius=ops['inner_neuropil_radius'],
            min_neuropil_pixels=ops['min_neuropil_pixels'],
            )
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        f_reg = BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], filename=ops['reg_file'])
        F, Fneu, _, _ = extraction.extract.extraction_wrapper(
            stat_manual, f_reg,
            cell_masks=manual_cell_masks,
            neuropil_masks=manual_neuropil_masks,
            device=device,
        )
        f_reg.close()

        np.save(os.path.join(plane_path, 'F_red'), F)
        np.save(os.path.join(plane_path, 'Fneu_red'), Fneu)
        # np.save(os.path.join(plane_path, 'F_redchan2_man'), F_chan2)
        # np.save(os.path.join(plane_path, 'F_red_neuchan2_man'), Fneu_chan2)  