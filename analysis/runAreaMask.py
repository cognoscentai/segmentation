from tqdm import tqdm
from sample_worker_seeds import sample_specs
import numpy as np 
import pickle as pkl
from PIL import Image, ImageDraw
from time import time 
from matplotlib import pyplot as plt
def create_tarea_mask(sample,objid):
    mega_mask = pkl.load(open('pixel_em/{}/obj{}/mega_mask.pkl'.format(sample,objid)))
    # Create masks for single valued tiles (so that they are more disconnected)
    from skimage import measure
    from matplotlib import _cntr as cntr
    tiles = [] # list of masks of all the tiles extracted
    tarea_mask = np.zeros_like(mega_mask)
    unique_tile_values = np.unique(mega_mask)
    tarea_lst = []
    for tile_value in unique_tile_values[1:]: #exclude 0
        blobs = mega_mask==tile_value
        blobs_labels = measure.label(blobs,background=0)
        for i in np.unique(blobs_labels)[1:]: 
            tile_mask = blobs_labels==i
            tile_pix = np.where(tile_mask==True)
            tiles.append(tile_mask)
	    tarea = mask_area(tile_mask)
            tarea_mask[tile_pix] = tarea
	    tarea_lst.append(tarea)
    outside_area  = np.product(np.shape(tarea_mask))-np.unique(tarea_mask).sum()
    tarea_mask[np.where(tarea_mask==0)]=outside_area
    tarea_lst.append(outside_area)
    pkl.dump(tarea_lst,open("pixel_em/{}/obj{}/tarea.pkl".format(sample,objid),'w'))
    pkl.dump(tarea_mask,open("pixel_em/{}/obj{}/tarea_mask.pkl".format(sample,objid),'w'))
    return tarea_mask
def plot_tarea_mask(tarea_mask):
    from matplotlib.colors import LogNorm
    plt.imshow(tarea_mask, norm=LogNorm(vmin=1, vmax=np.max(tarea_mask)))
    plt.colorbar()
def mask_area(mask):
    return len(np.where(mask)[0])
##############################################################################################
def neighbor_widx(wmap,source):
    x=source[0]
    y=source[1]
    return (x+1,y),(x,y+1),(x-1,y),(x,y-1)
def edge_neighbor_widx(wmap,source):
    x=source[0]
    y=source[1]
    valid_neighbors = []
    w,h = np.shape(wmap) 
    if x+1<w:
	valid_neighbors.append((x+1,y))
    if y+1<h:
	valid_neighbors.append((x,y+1))
    if x-1>=0:
	valid_neighbors.append((x-1,y))
    if y-1>=0:
	valid_neighbors.append((x,y-1))
    #return (x+1,y),(x,y+1),(x-1,y),(x,y-1)
    return valid_neighbors
def index_item(pix_lst,item):
    for i,pix in enumerate(pix_lst):
        if str(list(pix))==str(list(item)):
             return i 

def create_PixTiles(sample,objid,check_edges=False):
    wmap  = pkl.load(open("pixel_em/{}/obj{}/voted_workers_mask.pkl".format(sample,objid)))
    #print "wmap:",np.shape(wmap)
    tiles= []
    # Large outside tile 
    x,y = np.where(wmap==0)
    # print len(x), len(y)
    tiles.append(set(zip(x,y)))
    # print len(tiles)
    # Pixels voted by at least one worker
    x,y = np.where(wmap!=0)

    potential_pixs = np.array(zip(x,y))
    #Compute the votes on each pixel 
    votes = np.array([len(wmap[tuple(pix)]) for pix in potential_pixs])

    # print len(potential_pixs), len(potential_pixs[0])
    # print [v for v in votes if v > 5]

    # sort from lowest to highest number of votes  (low votes likely to be lone/small tiles)
    srt_idx = np.argsort(votes)
    # print srt_idx
    # print votes[srt_idx]
    srt_potential_pix = potential_pixs[srt_idx]

    # return
    tidx=1

    pixs_already_tiled = set()

    for source in srt_potential_pix:

        source = tuple(source)

        if source in pixs_already_tiled:
            continue
       
        pixs_already_tiled.add(source)
        tiles.append(set([source]))
        voted_workers = wmap[source]
        pidx =1
        potential_sources = set([source])
        checked_pixs = set()
        next_source = source

        while next_source is not None:
            at_least_one_connection=False
	    if (check_edges):
            neighbors = edge_neighbor_widx(wmap,next_source)
	    else: 
            neighbors = neighbor_widx(wmap,next_source)
            for neighbor in neighbors:
                if wmap[neighbor] == voted_workers:
                    tiles[tidx].add(neighbor)
                    # Remove added neighbor from potential pixels 
                    pixs_already_tiled.add(neighbor)
                    potential_sources.add(neighbor)

            checked_pixs.add(next_source)
            pixs_already_tiled.add(next_source)
            if len(potential_sources) > len(checked_pixs):
                next_source = (potential_sources - checked_pixs).pop()
            else:
                next_source = None
            pidx+=1

        tidx+=1 #moving onto the next tile
    pkl.dump(tiles,open("pixel_em/{}/obj{}/tiles.pkl".format(sample,objid),'w'))
 
if __name__=="__main__":
    import os
    object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
    for sample in tqdm(sample_specs.keys()):
        print sample 
        for objid in object_lst:
    	    print "objid:",objid
    	    #try:
            create_PixTiles(sample,objid,check_edges=True)
    	    #except(IOError):
            #	print "IO Error on:",sample, objid
