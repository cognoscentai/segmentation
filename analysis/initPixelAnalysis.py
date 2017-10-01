from PixelEM import *
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]

#print "Creating all worker and GT pixel masks (2-3 min)"
#for objid in object_lst:
#    create_all_gt_and_worker_masks(objid)

#from sample_worker_seeds import sample_specs
#sample_lst = sample_specs.keys()
#print "Creating megamask (aggregated mask over all workers in that sample) for all sample-objects [mega_mask.pkl, voted_workers_mask.pkl]"
#print "This might take a while"
#for sample in sample_lst:
#	for objid in object_lst:
#		print sample+":"+str(objid)
#		create_mega_mask(objid, PLOT=False, sample_name=sample)

from areaMask import * 
print "Creating area mask for all sample-objects"
for sample in tqdm(sample_specs.keys()):
        for objid in object_lst:
		print sample+":"+str(objid)
	        create_PixTiles(sample,objid,check_edges=True)		
