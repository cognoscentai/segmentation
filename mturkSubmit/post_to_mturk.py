#!flask/bin/python
import time
from glob import glob
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, NumberHitsApprovedRequirement #, Requirement
from boto.mturk.price import Price
from secret import SECRET_KEY,ACCESS_KEY,AMAZON_HOST
import os
from ADE20K_analysis_toolbox import load_ADE20K_info
import pandas as pd
#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
							 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
							 host=AMAZON_HOST)
print "Connected."
#frame_height in pixels
frame_height = 800

#Here, I create two sample qualifications
qualifications = Qualifications()
#qualifications.add(Requirement(MastersQualID,'DoesNotExist',required_to_preview=True))
qualifications.add(PercentAssignmentsApprovedRequirement(comparator="GreaterThan", integer_value="95"))
qualifications.add(NumberHitsApprovedRequirement(comparator="GreaterThan", integer_value="500"))

#Join object and image tables
img_info, object_tbl, BB_count_info = load_ADE20K_info()
BB_count_info = pd.read_csv("BB_count_tbl.csv") #override with local BB_count_tbl.csv
img_obj_tbl = object_tbl.merge(img_info,how="inner",left_on="image_id",right_on="id")

MAX_PER_OBJ = 40 
#This url will be the url of your application, with appropriate GET parameters
remaining_count  = 0
with open('ActiveHITs','a') as f:
	f.write('New batch created on : '+time.ctime())
	print "here"
	print glob("../web-app/app/static/ADE_train_*.png")
	for fname in glob("../web-app/app/static/ADE_train_*.png")[::-1]:
		img_name = fname.split('/')[-1].split('.')[0]
		print fname
		print img_name
		objId_lst = list(img_obj_tbl[img_obj_tbl.filename==img_name].object_id)
		print "objId_lst:",objId_lst
		for objId in objId_lst:
			print "objId:",objId
			url = "https://crowd-segmentation.herokuapp.com/segment/{0}/{1}/".format(img_name,objId)
			maxAssignment = MAX_PER_OBJ-int(BB_count_info[BB_count_info.id ==objId]["approved_BB_count"])
			print "maxAssignment:",maxAssignment
			remaining_count  +=maxAssignment
			if maxAssignment>0:
				questionform = ExternalQuestion(url, frame_height)
				create_hit_result = connection.create_hit(
					title="Segment the object on an image",
					description="We'll give you an image with a pointer to an object. You have to draw a bounding region around the boundary of the object in the image. There is 1 object per HIT. Our interface supports keyboard input for speed!",
					keywords=["segmentation", "perception", "image", "fast"],
					duration = 1800,
					max_assignments=maxAssignment,
					question=questionform,
					reward=Price(amount=0.05),
					lifetime=43200)#,
					#qualifications=qualifications)
				hit_id = str(create_hit_result[0].HITId)
				f.write(hit_id + "\n")
				print "Created HIT for img:{0}, objId:{1}: {2}".format(img_name,objId,hit_id)
print "Total Number of Remaining count:", remaining_count 
print "Total remaining cost:", remaining_count*0.06
