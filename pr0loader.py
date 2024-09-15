import json
import os
import time
import logging
from datetime import datetime
from typing import Any
import dotenv
import requests
from pymongo import MongoClient
from pymongo.collection import Collection
import pymongo.errors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)

# Global configuration
required_config_keys = ['ME', 'CONSENT', 'MONGODB_STRING', 'FILESYSTEM_PREFIX']
content_flags = 15  # Adjust as needed
http_max_tries = 100
http_timeout = 30


def get_collection(col_name: str) -> Collection:
    """
    Connects to the MongoDB database and returns the specified collection.

    Args:
        col_name (str): The name of the collection to retrieve.

    Returns:
        Collection: The MongoDB collection object.
    """
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    mydb = client["pr0loader"]
    return mydb[col_name]


def can_run():
    """
    Checks if the required configuration keys are present.

    Returns:
        bool: True if all required keys are present, False otherwise.
    """
    return all(key in config for key in required_config_keys)


def setup_session():
    """
    Sets up the requests session with the necessary cookies.

    Returns:
        requests.Session: The configured session object.
    """
    session = requests.Session()
    cookies = {
        'me': config['ME'],
        'euconsent-v2': config['CONSENT']
    }
    session.cookies.update(cookies)
    return session


def fetch_json_data(url, session):
    """
    Fetches JSON data from a given URL using the provided session.

    Args:
        url (str): The URL to fetch data from.
        session (requests.Session): The session object to use for the request.

    Returns:
        dict: The JSON data fetched from the URL.

    Raises:
        Exception: If the data cannot be fetched after max retries.
    """
    tries = 1
    while tries <= http_max_tries:
        try:
            logging.info(f"Fetching remote value try {tries} out of {http_max_tries} for {url}")
            response = session.get(url, timeout=http_timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            tries += 1
            logging.error(f"There was an error fetching remote data for {url}: {e}")
            time.sleep(tries)
    logging.error(f"Aborting fetching remote data after {http_max_tries} attempts for {url}")
    raise Exception(f"Failed to fetch data from {url}")


def db_pos(sort_order):
    """
    Retrieves the position (ID) from the database based on the sort order.

    Args:
        sort_order (list): The sort order for the query.

    Returns:
        int: The ID from the database, or -1 if not found.
    """
    try:
        doc = pr0_items_collection.find_one({}, projection={'id': 1}, sort=sort_order)
        if doc:
            return doc['id']
        else:
            return -1
    except pymongo.errors.PyMongoError as e:
        logging.error(f"Database error: {e}")
        return -1


def min_db_pos():
    """
    Retrieves the minimum ID from the database.

    Returns:
        int: The minimum ID, or -1 if not found.
    """
    return db_pos([('id', 1)])


def max_db_pos():
    """
    Retrieves the maximum ID from the database.

    Returns:
        int: The maximum ID, or -1 if not found.
    """
    return db_pos([('id', -1)])


def fetch_remote_value():
    """
    Fetches the highest ID value from the remote API.

    Returns:
        int: The highest remote ID value.
    """
    logging.info("Fetching data from remote to determine starting point")
    url = f"https://pr0gramm.com/api/items/get?flags={content_flags}"
    data = fetch_json_data(url, session)
    return data['items'][0]['id']


def determine_start():
    """
    Determines the starting point (current_id) for data fetching.

    Returns:
        int: The starting ID for data fetching.
    """
    min_id_in_db = min_db_pos()
    max_id_in_db = max_db_pos()
    logging.info(f"The DB currently has {min_id_in_db} as its lowest value")
    logging.info(f"The DB currently has {max_id_in_db} as its highest value")
    highest_remote_value = fetch_remote_value()
    logging.info(f"The remote currently has {highest_remote_value} as its highest value")
    if max_id_in_db == -1 or min_id_in_db == -1:
        # No local data
        logging.info("No data in DB found")
        return highest_remote_value
    if min_id_in_db != 1:
        return min_id_in_db
    return max_id_in_db


def fetch_info_for_item(item_id):
    """
    Fetches additional info for a specific item.

    Args:
        item_id (int): The ID of the item.

    Returns:
        dict: The JSON data containing item info.
    """
    url = f"https://pr0gramm.com/api/items/info?itemId={item_id}&flags={content_flags}"
    return fetch_json_data(url, session)


def process_media_metadata(json_data):
    """
    Processes media metadata and inserts it into the database.

    Args:
        json_data (dict): The JSON data containing media items.
    """
    for item in json_data['items']:
        add_info = fetch_info_for_item(item['id'])
        if 'comments' in add_info:
            item['comments'] = add_info['comments']
        if 'tags' in add_info:
            item['tags'] = add_info['tags']
        try:
            pr0_items_collection.insert_one(item)
            logging.info(f"Written item {item['id']} to database")
        except pymongo.errors.PyMongoError as e:
            logging.error(f"Failed to insert item {item['id']} into database: {e}")


def get_fs_prefix():
    """
    Retrieves the filesystem prefix from the configuration.

    Returns:
        str: The filesystem prefix with a trailing slash.
    """
    config_value = str(config['FILESYSTEM_PREFIX'])
    if not config_value.endswith("/"):
        config_value += "/"
    return config_value


def download_medias(json_data):
    """
    Downloads media files based on the JSON data.

    Args:
        json_data (dict): The JSON data containing media items.
    """
    for item in json_data['items']:
        media_name = item['image']
        fs_prefix = get_fs_prefix()
        remote_media_prefix = "https://img.pr0gramm.com/"
        local_file = os.path.join(fs_prefix, media_name)
        logging.info(f"The media name would be: {local_file}")
        url = remote_media_prefix + media_name

        tries = 1
        while tries <= http_max_tries:
            try:
                logging.info(f"Fetching remote value try {tries} out of {http_max_tries} for {url}")
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                response = session.get(url, timeout=http_timeout, stream=True)
                response.raise_for_status()
                size = 0
                with open(local_file, 'wb') as file_handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_handle.write(chunk)
                            size += len(chunk)
                logging.info(f"Written data: {size} bytes for {media_name}")
                break  # Exit the while loop upon success
            except requests.HTTPError as e:
                if 400 <= e.response.status_code < 500:
                    logging.error(f"HTTPError {e.response.status_code}: {e.response.reason}")
                    break  # Client error, do not retry
                else:
                    tries += 1
                    logging.error(f"There was an error fetching remote data for {url}: {e}")
                    time.sleep(tries)
            except Exception as e:
                tries += 1
                logging.error(f"An unexpected exception was caught: {e}")
                time.sleep(tries)
        else:
            logging.error(f"Aborting fetching remote data after {http_max_tries} attempts for {url}")


def get_next_current_id(json_data):
    """
    Retrieves the next current ID from the JSON data.

    Args:
        json_data (dict): The JSON data containing media items.

    Returns:
        int or None: The next ID to process, or None if not available.
    """
    items = json_data.get('items', [])
    if items:
        return items[-1]['id']
    else:
        return None


def main():
    """
    The main function that orchestrates the data fetching and processing.
    """
    global config, session, pr0_items_collection

    logging.info("Reading configuration")
    config = {
        **dotenv.dotenv_values(".env"),
        **os.environ
    }

    logging.info("Checking if configuration satisfies minimal config")
    if not can_run():
        logging.error(
            f"It's required to have at least {', '.join(required_config_keys)} set in your env to run this program"
        )
        exit(1)

    logging.info("Setting up session and cookies")
    session = setup_session()

    logging.info("Preparing DB collections")
    pr0_items_collection = get_collection("pr0items")

    logging.info("Determine starting position")
    current_id = determine_start()
    logging.info(f"About to read from id {current_id}")

    # Start reading from remote
    cancel = False

    while not cancel:
        logging.info(f"Fetching data from remote starting with id {current_id}")
        url = f"https://pr0gramm.com/api/items/get?older={current_id}&flags={content_flags}"
        json_data = fetch_json_data(url, session)
        process_media_metadata(json_data)
        download_medias(json_data)
        next_id = get_next_current_id(json_data)
        if next_id is None or next_id >= current_id or next_id <= 1:
            logging.info("No more items to process or invalid next ID.")
            cancel = True
        else:
            current_id = next_id
        time.sleep(1)  # Sleep to respect API rate limits


if __name__ == "__main__":
    main()