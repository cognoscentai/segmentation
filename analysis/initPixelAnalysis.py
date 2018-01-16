from PixelEM import *
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]

#print "Creating all worker and GT pixel masks (2-3 min)"
#for objid in object_lst:
#    create_all_gt_and_worker_masks(objid)

from sample_worker_seeds import sample_specs
sample_lst = sample_specs.keys()
#print "Creating megamask (aggregated mask over all workers in that sample) for all sample-objects [mega_mask.pkl, voted_workers_mask.pkl]"
#print "This might take a while (~2hrs)"
#for sample in sample_lst:
#	for objid in object_lst:
#		print sample+":"+str(objid)
#		create_mega_mask(objid, PLOT=False, sample_name=sample)

#print "Creating MV mask (should take 5 min)"
#for sample in sample_lst:
#	for objid in object_lst:
#		print sample+":"+str(objid)
#		create_MV_mask(sample, objid)

#from areaMask import * 
#print "Creating area mask for all sample-objects"
#print "This will also take a while (~5hrs)"
#for sample in tqdm(sample_lst):
#        for objid in object_lst:
#		if os.path.exists("pixel_em/{}/obj{}/tiles.pkl".format(sample,objid)):
#			print sample+":"+str(objid)+" already exist"
#		else:
#			print sample+":"+str(objid)
#	        	create_PixTiles(sample,objid,check_edges=True)	

############################################################
## DEBUG PIXTILE OUTPUT (VISUALLY INSPECT)
# def tiles2AreaMask(sample,objid):
#    tiles = pkl.load(open("pixel_em/{}/obj{}/tiles.pkl".format(sample,objid)))
#    mega_mask = pkl.load(open("pixel_em/{}/obj{}/mega_mask.pkl".format(sample,objid)))
#    tarea = [len(t) for t in tiles]
#    mask = np.zeros_like(mega_mask)
#    for tidx in range(len(tiles)):
#        for i in list(tiles[tidx]):
#            mask[i]=tarea[tidx]
#    return mask

#mask = tiles2mask("5workers_rand0",1)
#plt.figure()
#plt.imshow(mask)
#plt.title("Tile index map")
#plt.colorbar()
############################################################

# Check number of object that have completed their full run by :
# ls pixel_em/*/obj*/EM_prj_iter2_thresh-4.json |wc -l
# ls pixel_em/*/obj*/GT_EM_prj_iter2_thresh-4.json |wc -l
#ls pixel_em/*/obj*/GTLSA_EM_prj_iter1_thresh-4.json |wc -l
# Use  submitPixelEM.sh to submit all the jobs in parallel for different samples independently
'''
sample = sys.argv[1]
print sample   
#for sample in tqdm(sample_specs.keys()):

for objid in object_lst[::-1]:
	print sample+":"+str(objid)
	#if True: 
	#	thresh=-4
	for thresh in [-4,-2,0,2,4]:
		do_EM_for(sample, objid,thresh=thresh,rerun_existing=False,compute_PR_every_iter=True,exclude_isovote=False,num_iterations=3)
                do_GT_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=True,compute_PR_every_iter=True, num_iterations=3)
                do_GT_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=False,compute_PR_every_iter=True, num_iterations=3)
	     	do_GTLSA_EM_for(sample, objid,thresh=thresh,rerun_existing=False,compute_PR_every_iter=True,exclude_isovote=True, num_iterations=3)	
                do_GTLSA_EM_for(sample, objid,thresh=thresh,rerun_existing=False,compute_PR_every_iter=True,exclude_isovote=False, num_iterations=3)
'''

'''
# Compiled PRJ to :/home/jlee782/segmentation/analysis/pixel_em/<algoname>_full_PRJ_table.csv
print "Compiling the output from .json to one single csv file for each algo (should take ~1min)" 
algorithms = ["GTLSA","isoGTLSA","GT","isoGT","basic"]
for algo in algorithms: 
	compile_PR(mode=algo,ground_truth=False)
'''
