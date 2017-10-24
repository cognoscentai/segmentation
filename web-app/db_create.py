#!flask/bin/python
from migrate.versioning import api
from config import SQLALCHEMY_DATABASE_URI,SQLALCHEMY_MIGRATE_REPO
from app import db
import os.path
# Creating all database tables and schema
db.create_all()
if not os.path.exists(SQLALCHEMY_MIGRATE_REPO):
    api.create(SQLALCHEMY_MIGRATE_REPO, 'database repository')
    api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO)
else:
    api.version_control(SQLALCHEMY_DATABASE_URI, SQLALCHEMY_MIGRATE_REPO, api.version(SQLALCHEMY_MIGRATE_REPO))

# Loading object and object_location data into database 
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("postgresql://segment@localhost:5432")
# Drop image, object, object_location table to avoid foreign key constraints errors
engine.execute("DROP table image, object, object_location CASCADE;")
for fname in ["image","object","object_location"]:
    df = pd.read_csv("../data/{}.csv".format(fname),skip_footer=True)
    df.to_sql(name=fname, con=engine, if_exists = 'replace', index=False)


