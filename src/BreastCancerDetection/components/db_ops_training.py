from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json

from BreastCancerDetection.entity import (DatabaseOperationsCredentials)


class DatabaseOperations:
    def __init__(self, 
                credentials: DatabaseOperationsCredentials):
        self.credentials = credentials
    
    def database_connection_establishment(self):

        ASTRA_TOKEN_PATH = self.credentials.ASTRA_TOKEN_PATH
        ASTRA_DB_SECURE_BUNDLE_PATH = self.credentials.ASTRA_DB_SECURE_BUNDLE_PATH

        with open(ASTRA_TOKEN_PATH, "r") as f:
            creds = json.load(f)
            ASTRA_DB_APPLICATION_TOKEN = creds["token"]

        cluster = Cluster(
            cloud={
                "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
            },
            auth_provider=PlainTextAuthProvider(
                "token",
                ASTRA_DB_APPLICATION_TOKEN,
            ),
        )

        session = cluster.connect()

        row = session.execute("select release_version from system.local").one()
        if row:
            print(row[0])
        else:
            print("An error occurred.")

