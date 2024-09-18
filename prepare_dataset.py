import os
import logging
import re
import csv
import datetime
import dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
import pymongo.errors
from typing import List, Dict
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] --- %(levelname)s --- %(message)s',
    datefmt='%m/%d/%Y, %H:%M:%S'
)

# Global Constants and Configurations
DEVELOP_MODE = True  # Set to True for development mode, False for normal operation
VALID_TAG_REGEX = re.compile(r'^[a-zA-Z0-9]+$')  # Regular expression to match valid tags (alphanumeric only)
IMAGE_EXTENSIONS_REGEX = r'\.(jpg|jpeg|png)$'  # Regex pattern to match image extensions
NSFW_TAGS_SET = {'nsfw', 'nsfl', 'nsfp'}  # Set of tags to identify NSFW content
MINIMUM_VALID_TAGS = 5  # Minimum number of valid tags required after processing

def get_mongo_collection(collection_name: str, config: dict) -> Collection:
    """
    Connect to the MongoDB database and return the specified collection.
    """
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    database = client["pr0loader"]
    return database[collection_name]

def validate_config(config: dict) -> bool:
    """
    Check if the required configuration keys are present in the config dictionary.
    """
    required_keys = ['MONGODB_STRING']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {', '.join(missing_keys)}")
        return False
    return True

def load_config() -> dict:
    """
    Load configuration from environment variables and .env file.
    """
    return {**dotenv.dotenv_values(".env"), **os.environ}

def generate_timestamped_filename(prefix: str = 'output', extension: str = 'csv') -> str:
    """
    Generate a timestamped filename with the given prefix and extension.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{timestamp}_{prefix}.{extension}'

def build_query() -> dict:
    """
    Build the MongoDB query to filter the desired documents.
    """
    query = {
        'image': {'$regex': IMAGE_EXTENSIONS_REGEX, '$options': 'i'},
        '$expr': {'$gte': [{'$size': '$tags'}, MINIMUM_VALID_TAGS]}
    }
    return query

def fetch_documents(collection: Collection, query: dict) -> List[dict]:
    """
    Fetch documents from the MongoDB collection based on the query.
    """
    try:
        if DEVELOP_MODE:
            logging.info("DEVELOP_MODE is ON: Limiting results to 10 documents")
            return list(collection.find(query).limit(10))
        else:
            return list(collection.find(query))
    except pymongo.errors.PyMongoError as e:
        logging.error(f"Database error occurred: {e}")
        return []

def process_tags(tags: List[dict]) -> Dict[str, any]:
    """
    Process the tags by filtering, removing NSFW tags, and validating tag names.
    Returns a dictionary with NSFW flags and a list of valid tags.
    """
    # Initialize NSFW flags
    is_nsfw_present = is_nsfl_present = is_nsfp_present = False

    # Check for NSFW tags and remove them
    filtered_tags = []
    for tag in tags:
        tag_name = tag.get('tag', '').lower()
        if tag_name in NSFW_TAGS_SET:
            if tag_name == 'nsfw':
                is_nsfw_present = True
            elif tag_name == 'nsfl':
                is_nsfl_present = True
            elif tag_name == 'nsfp':
                is_nsfp_present = True
            continue  # Skip NSFW tags
        else:
            filtered_tags.append(tag)

    # Validate tags to include only alphanumeric characters
    valid_tags = []
    for tag in filtered_tags:
        tag_name = tag.get('tag', '')
        if VALID_TAG_REGEX.match(tag_name):
            valid_tags.append(tag)
        else:
            logging.debug(f"Excluding tag '{tag_name}' as it contains invalid characters.")

    # Return NSFW flags and valid tags
    return {
        'is_nsfw_present': is_nsfw_present,
        'is_nsfl_present': is_nsfl_present,
        'is_nsfp_present': is_nsfp_present,
        'valid_tags': valid_tags
    }

def prepare_output_item(document: dict) -> Dict[str, any]:
    """
    Prepare the output item by processing the document and its tags.
    Returns a dictionary suitable for CSV writing or None if the document should be skipped.
    """
    output_item = {
        'id': document.get('id', ''),
        'image': document.get('image', '')
    }

    tags = document.get('tags', [])
    if not isinstance(tags, list):
        return None  # Skip if 'tags' is not a list

    tag_processing_result = process_tags(tags)
    valid_tags = tag_processing_result['valid_tags']

    # Ensure there are at least MINIMUM_VALID_TAGS valid tags
    if len(valid_tags) < MINIMUM_VALID_TAGS:
        return None  # Skip this document

    # Set NSFW flags
    output_item['is_nsfw'] = 'true' if tag_processing_result['is_nsfw_present'] else 'false'
    output_item['is_nsfl'] = 'true' if tag_processing_result['is_nsfl_present'] else 'false'
    output_item['is_nsfp'] = 'true' if tag_processing_result['is_nsfp_present'] else 'false'

    # Sort tags by confidence
    sorted_tags = sorted(valid_tags, key=lambda x: x.get('confidence', 0), reverse=True)

    # Take the top MINIMUM_VALID_TAGS tags
    top_tags = sorted_tags[:MINIMUM_VALID_TAGS]

    # Add tags and confidences to output_item
    for idx, tag in enumerate(top_tags, start=1):
        output_item[f'tag{idx}'] = tag.get('tag', '')
        output_item[f'confidence{idx}'] = tag.get('confidence', '')

    return output_item

def write_to_csv(filename: str, fieldnames: List[str], data: List[Dict[str, any]]):
    """
    Write the data to a CSV file with the given filename and fieldnames.
    """
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for item in data:
                csv_writer.writerow(item)
        logging.info(f"CSV file saved as: {filename}")
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")

def pretty_print_data(fieldnames: List[str], data: List[Dict[str, any]]):
    """
    Pretty print the data using the tabulate library.
    """
    if DEVELOP_MODE and data:
        results_list = [[item.get(field, '') for field in fieldnames] for item in data]
        print("\nCSV Output:")
        print(tabulate(results_list, headers=fieldnames, tablefmt='grid'))

def main():
    """
    Main function to execute the data extraction and processing pipeline.
    """
    logging.info("Starting the MongoDB data reader script")

    # Load and validate configuration
    config = load_config()
    if not validate_config(config):
        logging.error("Please ensure MONGODB_STRING is set in your configuration.")
        exit(1)

    # Connect to MongoDB and get the collection
    mongo_collection = get_mongo_collection("pr0items", config)

    # Build the query and fetch documents
    query = build_query()
    documents = fetch_documents(mongo_collection, query)

    if not documents:
        logging.info("No documents found matching the query.")
        return

    # Prepare the CSV fieldnames
    fieldnames = ['id', 'image', 'is_nsfw', 'is_nsfl', 'is_nsfp']
    for i in range(1, MINIMUM_VALID_TAGS + 1):
        fieldnames.append(f'tag{i}')
        fieldnames.append(f'confidence{i}')

    # Process documents and collect output items
    output_data = []
    for document in documents:
        output_item = prepare_output_item(document)
        if output_item:
            output_data.append(output_item)

    if not output_data:
        logging.info("No valid documents found after processing.")
        return

    # Generate the CSV filename
    csv_filename = generate_timestamped_filename()

    # Write output data to CSV
    write_to_csv(csv_filename, fieldnames, output_data)

    # Pretty print the data if in development mode
    pretty_print_data(fieldnames, output_data)

    logging.info("MongoDB data reader script has completed")

if __name__ == "__main__":
    main()