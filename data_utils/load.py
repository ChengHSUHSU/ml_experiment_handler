

import pickle


def read_pickle(file_name, base_path='./datasets'):
    with open(f'{base_path}/{file_name}', 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as err:
            print('fail to load pickle file.')
            return None

        
        
def download_blob(bucket_name, source_blob_name, destination_file_name):
    from google.cloud import storage
    """Downloads a blob from the bucket."""
    try:
        print('Starting Download.....')
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f'Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}.')
    except Exception as e:
        print(f'Downloaded storage object {source_blob_name} failed.')
        raise e
        
        
        
        