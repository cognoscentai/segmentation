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
#for sample in tqdm(sample_specs.keys()):
#        for objid in object_lst:
#		if os.path.exists("pixel_em/{}/obj{}/tiles.pkl".format(sample,objid)):
#			print sample+":"+str(objid)+" already exist"
#		else:
#			print sample+":"+str(objid)
#	        	create_PixTiles(sample,objid,check_edges=True)	

for sample in tqdm(sample_specs.keys()):
	for objid in object_lst:
		print sample+":"+str(objid)
		for thresh in [-4,-2,0,2,4]:
			do_EM_for(sample, objid,thresh=thresh,compute_PR_every_iter=False,exclude_isovote=False)
			do_GTLSA_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=True, num_iterations=3)
                	do_GTLSA_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=False, num_iterations=3)
                	do_GT_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=True,compute_PR_every_iter=True, num_iterations=3)
                	do_GT_EM_for(sample, objid,thresh=thresh,rerun_existing=False,exclude_isovote=False,compute_PR_every_iter=True, num_iterations=3)			
