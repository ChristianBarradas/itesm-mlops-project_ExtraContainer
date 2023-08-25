import pandas as pd
from custom_loggin import CustomLogger

FILE_LOG_DIR = 'load/load_data.log'
logger = CustomLogger(__name__, FILE_LOG_DIR).logger    
logger.info("load data")
class DataRetriever:
    logger.info("DataRetriever")
    """
    A class for retrieving data from a given URL and processing it for further analysis.

    Parameters:
        url (str): The URL from which the data will be loaded.


    Attributes:
        url (str): The URL from which the data will be loaded.


    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = DataRetriever(URL, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """
    DROP_COLS = ['id', 'zipcode', 'date']
    RETRIEVED_DATA = '/data/kc_house_data.csv'  # File name for the retrieved data.
    logger.info(f"drop columns:{[DROP_COLS]}")
    

    def __init__(self, url, data_path):
        self.url = url
        self.DATASETS_DIR = data_path

    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.
        Returns:
            str: A message indicating the location of the stored data.
        """
        # Loading data from specific URL
        data = pd.read_csv(self.DATASETS_DIR)
        # Drop irrelevant columns
        data.drop(self.DROP_COLS, axis=1, inplace=True)
        return data
        # Usage Example:
        # URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
        # data_retriever = DataRetriever(URL)
        # result = data_retriever.retrieve_data()
        # print(result)
