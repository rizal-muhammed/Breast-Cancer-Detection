from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
import shutil

from BreastCancerDetection.logging import logger
from BreastCancerDetection.utils import common
from BreastCancerDetection.entity import (DataBaseOperationsTrainingConfig,
                                          DataBaseOperationsTrainingCredentials,
                                          DataBaseOperationsTrainingParams)


class DatabaseOperations:
    def __init__(self,
                 config: DataBaseOperationsTrainingConfig,
                 credentials: DataBaseOperationsTrainingCredentials,
                 params: DataBaseOperationsTrainingParams) -> None:
        self.config = config
        self.credentials = credentials
        self.params = params

        self.cluster = None
        self.session = None

        common.create_directories([self.config.root_dir, self.config.good_raw, self.config.bad_raw])

    def get_session(self,):
        """
            This method establishes a session with Astra Database Keyspace specified by the 
            parameter 'ASTRA_DB_KEYSPACE' and return the session.

            If a session is already running, corresponding session is returned.

            Parameters
            ----------
            None

            Returns
            -------
            Returns the session to the corresponding Keyspace specified in the params.

            Raises
            ------
            ConnectionError
            Exception
        
        """
        try:
            ASTRA_DB_SECURE_BUNDLE_PATH = Path(self.credentials.ASTRA_DB_SECURE_BUNDLE_PATH)
            ASTRA_TOKEN_PATH = Path(self.credentials.ASTRA_TOKEN_PATH)

            if self.session is None:
                logger.info(f"""[get_session] Creating session""")

                cloud_config= {
                'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
                }

                with open(ASTRA_TOKEN_PATH) as f:
                    secrets = json.load(f)

                CLIENT_ID = secrets["clientId"]
                CLIENT_SECRET = secrets["secret"]

                auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
                self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, protocol_version = 4)

                self.session = self.cluster.connect(self.params.ASTRA_DB_KEYSPACE)
            else:
                logger.info(f"""get_session] Reusing session""")

            return self.session
        
        except Exception as e:
            logger.exception(f"""Exception at 'get_session'. 
                             Exception message: {str(e)}""")
            raise e
    
    def shutdown_driver(self):
        """
            This method shutdown the cluster and session

            This method can be used at the exit for shutting down the cluster and session, thus closing the connection.

            Parameters
            ----------
            None

            Returns
            -------
            None.

            Raises
            ------
            Exception
        
        """
        try:
            if self.session is not None:
                logger.info(f"[shutdown_driver] Closing connection")
                self.cluster.shutdown()
                self.session.shutdown()
                logger.info(f"Database Connection closed successfully.")
        
        except Exception as e:
            logger.exception(f"""Exception while closing the connection.
                             Exception message: {str(e)}""")
            raise e

    def get_cassandra_cluster_info(self):
        try:
            self.session = self.get_session()

            cassandra_output_1 = self.session.execute("SELECT cluster_name, release_version FROM system.local")
            for cassandra_row in cassandra_output_1:
                output_message = "Connected to " + str(cassandra_row.cluster_name) + " and it is running " + str(cassandra_row.release_version)
                logger.info(output_message)
        
        except Exception as e:
            logger.exception(f"""Something went wrong. 
                             Exception message: {str(e)}""")
            raise e
    
    def create_table_db(self, ):
        """
            This method creates table specified in the params (if not exists) in the given database, 
            which  will be used to insert the good data after raw data validation

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> creating table(if not exists) for trainining files... <<<<<<<""")

            self.session = self.get_session()
            self.get_cassandra_cluster_info()

            table_name = self.params.table_name
            column_names = self.params.column_names
            query = f"SELECT table_name FROM system_schema.tables WHERE keyspace_name = '{self.session.keyspace}' AND table_name = '{table_name}'"
            result = self.session.execute(query)

            if result.one():
                logger.info(f"""Table '{table_name}' already exists. Adding columns if not exists...""")

                existing_columns = set()
                try:
                    result = self.session.execute(f"SELECT column_name FROM system_schema.columns WHERE keyspace_name = '{self.session.keyspace}' AND table_name = '{table_name}'")
                    existing_columns = {row.column_name for row in result}

                    for column, data_type in column_names.items():
                        # if table already exists, and the column doesn't exists, then add the column
                        if column not in existing_columns:
                            alter_table_query = f"ALTER TABLE {self.session.keyspace}.{table_name} ADD {column} {data_type}"
                            self.session.execute(alter_table_query)
                            logger.info(f"Column {column} added to table {table_name}.")
                except Exception as e:
                    logger.exception(f"Exception while adding columns. Exception message: {str(e)}")
                    raise e 
            else:
                try:
                    # table doesn't exists, so create table
                    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
                    for i, (column, data_type) in enumerate(column_names.items()):
                        if i == 0:
                            create_table_query += f"{column} {data_type} PRIMARY KEY, "
                        else:
                            create_table_query += f"{column} {data_type}, "

                    create_table_query = create_table_query.rstrip(', ')
                    create_table_query += ");"
                    logger.info(create_table_query)

                    try:
                        self.session.execute(create_table_query)
                        logger.info(f"""Table '{str(table_name)}' created successfully.""")
                    except Exception as e:
                        logger.exception(f"""Error creating table. Error message: {str(e)}""")

                except Exception as e:
                    logger.exception(f"Exception while creating table. Exception message: {str(e)}")
                    raise e
                
        
        except Exception as e:
            logger.exception(f"""Exception in 'create_table_db' method of 'DatabaseOperations' class.
                             Exception message : {str(e)}""")
            raise e
    
    def insert_into_table_good_data(self, ):
        """
            This method inserts the data from good_raw files to the given database table

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> insertion of good data into database table for trainining files started... <<<<<<<""")
            
            self.session = self.get_session()
            self.get_cassandra_cluster_info()

            all_items = os.listdir(self.config.good_raw)  # all items in the good_raw directory
            # listing only the files in the good_raw directory
            only_files = [item for item in all_items if os.path.isfile(os.path.join(self.config.good_raw, item)) and item != ".DS_Store"]

            if len(only_files) > 0:
                # opening good data files one be one and inserting the data into the given 
                for file in only_files:
                    try:
                        file_path = os.path.join(self.config.good_raw, file)
                        df = pd.read_csv(file_path)
                        df = df.fillna(np.nan)

                        if(list(self.params.column_names.keys()) == list(df.columns)):
                            for index, row in df.iterrows():
                                columns = ", ".join(row.index)
                                values = ", ".join(["%s" for _ in range(len(row))])

                                insert_query = f"INSERT INTO {str(self.params.table_name)} ({columns}) VALUES ({values})"
                                params = tuple(row)

                                self.session.execute(insert_query, params)
                                if index % 100 == 0:
                                    logger.info(f"Inserted {index} rows into the database.")
                        else:
                            logger.error(f"Could not insert into the table {str(self.params.table_name)}")
                            shutil.move(os.path.join(self.config.good_raw, file), self.config.bad_raw)
                            logger.info(f"""Since insertion failed, the file '{str(file)}' moved to 'bad_raw' data successfully""")

                    except Exception as e:
                        logger.exception( f"Exception occured while inserting data into the table : {str(e)}")
                        shutil.move(os.path.join(self.config.good_raw, file), self.config.bad_raw)
                        raise e
            
                logger.info(f"All the 'good_raw' files are inserted successfully into the 'good_raw' table")
            else:
                logger.info("Good raw folder is empty. There is nothing to insert.")
        
        except Exception as e:
            logger.exception( f"Exception occured while inserting data into the table : {str(e)}")
            raise e
        
    def export_data_from_table_into_csv(self, ):
        """
            This method exports data from the database into a csv file

            Parameters
            ----------
            None

            Returns
            -------
            None

            Raises
            ------
            Exception
        
        """
        try:
            logger.info(f""">>>>>>> exporting good data into csv file from training database... <<<<<<<""")

            self.session = self.get_session()
            self.get_cassandra_cluster_info()

            table_name = self.params.table_name
            key_space = self.params.ASTRA_DB_KEYSPACE

            query = f"SELECT * FROM {key_space}.{table_name}"
            result = self.session.execute(query)

            df = pd.DataFrame(result, columns=result.column_names)  # creating a pandas dataframe of data

            common.create_directories([self.config.root_dir])
            filepath = os.path.join(self.config.root_dir, self.config.file_name)

            df.to_csv(filepath, index=False, header=True)  # exporting to csv file
  
            logger.info(f"Input data exported to csv successfully")
        
        except Exception as e:
            logger.exception(f"Input data exporting to csv is failed, Error : {str(e)}")
            raise e
        
        finally:
            self.shutdown_driver()


