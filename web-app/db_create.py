#!flask/bin/python
from migrate.versioning import api
from config import SQLALCHEMY_DATABASE_URI,SQLALCHEMY_MIGRATE_REPO
from app import db
import os.path

# db.engine.execute("CREATE USER segment WITH CREATEDB CREATEROLE;")
# db.engine.execute("ALTER USER segment WITH SUPERUSER;")
# db.engine.execute("CREATE DATABASE segment;")
db.engine.execute("DROP schema public cascade; CREATE schema public;")
#Creating all database tables and schema
db.create_all()
# if not os.path.exists(SQLALCHEMY_MIGRATE_REPO):
#     api.create(SQLALCHEMY_MIGRATE_REPO, 'database repository')
#     api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
# else:
#     api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO, api.version(SQLALCHEMY_MIGRATE_REPO))

import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("postgresql://segment@localhost:5432")
# For object identification to populate object_location , you only need to drop and readd the image table
# engine.execute("DROP table image  CASCADE;") 
# for fname in ["image"]:#,"object","object_location"]:
#     df = pd.read_csv("../ADE20K_data_info/{}.csv".format(fname))
#     df.to_sql(name=fname, con=engine, if_exists = 'replace', index=False)

#For actual segmentation task, Load and Drop image, object, object_location table to avoid foreign key constraints errors
engine.execute("DROP table image, object, object_location  CASCADE;")
for fname in ["image","object","object_location"]:
    df = pd.read_csv("ADE20K_data_info/{}.csv".format(fname))
    df.to_sql(name=fname, con=engine, if_exists = 'replace', index=False)


