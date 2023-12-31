import http.cookiejar
import json
import os
import urllib.request
from datetime import datetime
from typing import Any, Mapping
from urllib.error import URLError, HTTPError
import time
import dotenv
from pymongo import MongoClient
from pymongo.database import Database

# global config
required_config_keys = ['ME', 'CONSENT', 'MONGODB_STRING', 'FILESYSTEM_PREFIX']
# read filter settings (nsfl, nsfw, nsfp)
content_flags = 15  # this needs to be calculated later
http_max_tries = 100
http_timeout = 30


def get_collection(col_name):
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    mydb: Database[Mapping[str, Any] | Any] = client["pr0loader"]
    mycol = mydb[col_name]
    return mycol


def concat(message):
    _result = ''
    for item in message:
        _result = _result + " " + str(item)
    return _result


def log(*message):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("[", date_time, "] --- " + concat(message))


# read start pointer from file (e.g. highest ID of image loaded
def db_pos(_sort):
    _filter = {}
    project = {
        'id': 1
    }
    sort = _sort
    limit = 1
    max_value: -1
    try:
        max_value = pr0_items_collection.find(filter=_filter, projection=project, sort=sort, limit=limit)[0]['id']
    except:
        max_value = -1
    return max_value


def min_db_pos():
    _sort = list({
                     'id': 1
                 }.items())

    return db_pos(_sort)


def max_db_pos():
    _sort = list({
                     'id': -1
                 }.items())

    return db_pos(_sort)


def fetch_remote_value():
    log("fetching data from remote to determine starting point")
    return fetch_json_data("https://pr0gramm.com/api/items/get&flags=" + str(15))['items'][0]['id']


def determine_start():
    # It's either to be fetched from remote or the lowest ID in the DB
    min_id_in_db = min_db_pos()
    max_id_in_db = max_db_pos()
    log("the db currently has", min_id_in_db, "as their lowest value")
    log("the db currently has", max_id_in_db, "as their highest value")
    highest_remote_value = fetch_remote_value()
    log("the remote currently has", highest_remote_value, "as their highest value")
    if max_id_in_db == -1 | min_id_in_db == -1:
        # no local data
        log("no data in db found")
        return highest_remote_value
    if min_id_in_db != 1:
        return min_id_in_db
    return max_id_in_db


def fetch_info_for_item(_id):
    _json_data = fetch_json_data(
        "https://pr0gramm.com/api/items/info?itemId=" + str(_id) + "&flags=" + str(content_flags))
    return _json_data


def fetch_json_data(url):
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    tries = 1
    successful = False
    while not successful:
        if tries > http_max_tries:
            log("Aborting fetching remote data since more than http_max_tries for", url)
            raise URLError
        try:
            log("Fetching remote value try", tries, "out of", http_max_tries, "for", url)
            result = opener.open(url, timeout=http_timeout)
            data = result.read()
            encoding = result.info().get_content_charset('utf-8')
            _json_data = json.loads(data.decode(encoding))
            opener.close()
            successful = True
            return _json_data
        except Exception as e:
            tries += 1
            log("There was an error fetching remote data for", url)
            time.sleep(tries)


def can_run():
    global required_config_keys
    _can_run = True
    for key in required_config_keys:
        if key not in config:
            _can_run = False
    return _can_run


def setup_cookie_jar():
    _jar = http.cookiejar.CookieJar()
    me_cookie = http.cookiejar.Cookie(version=0, name='me', value=config['ME'], port=None, port_specified=False,
                                      domain='pr0gramm.com', domain_specified=False, domain_initial_dot=False, path='/',
                                      path_specified=True, secure=True, expires=None, discard=True, comment=None,
                                      comment_url=None, rest={'HttpOnly': None}, rfc2109=False)
    consent_cookie = http.cookiejar.Cookie(version=0, name='euconsent-v2', value=config['CONSENT'], port=None,
                                           port_specified=False,
                                           domain='pr0gramm.com', domain_specified=False, domain_initial_dot=False,
                                           path='/',
                                           path_specified=True, secure=True, expires=None, discard=True, comment=None,
                                           comment_url=None, rest={'HttpOnly': None}, rfc2109=False)
    _jar.set_cookie(me_cookie)
    _jar.set_cookie(consent_cookie)
    return _jar


def process_media_metadata(_json_data):
    global current_id, cancel
    for item in _json_data['items']:
        add_info = fetch_info_for_item(item['id'])
        # print(item)
        if 'comments' in add_info:
            item['comments'] = add_info['comments']
        if 'tags' in add_info:
            item['tags'] = add_info['tags']
        result = pr0_items_collection.insert_one(item)
        # print(result)
        current_id = item['id']
        log("Written item", current_id, "to database")


def get_fs_prefix():
    config_value = str(config['FILESYSTEM_PREFIX'])
    if not config_value.endswith("/"):
        config_value += "/"
    return config_value


def download_medias(_json_data):
    for item in _json_data['items']:
        media_name = item['image']
        fs_prefix = get_fs_prefix()
        remote_media_prefix = "https://img.pr0gramm.com/"
        local_file = fs_prefix + media_name
        log("the media name would be:", local_file)
        _url = remote_media_prefix + media_name

        tries = 1
        successful = False
        while not successful:
            if tries > http_max_tries:
                log("Aborting fetching remote data since more than http_max_tries for", _url)
                raise Exception
            try:
                log("Fetching remote value try", tries, "out of", http_max_tries, "for", _url)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                file_handle = open(local_file, 'wb')
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
                result = opener.open(_url, timeout=http_timeout)
                # no need to retry over and over again in case of >400 http response code
                if result.status > 400:
                    successful = True
                    log("Server error downloading", _url)
                    break
                size = 0
                while True:
                    data = result.read(10000)
                    if len(data) < 1:
                        break
                    file_handle.write(data)
                    size = size + len(data)
                opener.close()
                file_handle.close()
                log("written data: ", size, "bytes for ", media_name)
                successful = True
            except HTTPError as e:  # while Exception is not nice, we want to catch all possible exceptions
                if 400 < e.code < 500:
                    log("There was an HTTPError > 400", e.reason)
                    break
                tries += 1
                log("There was an error fetching remote data for", _url)
                time.sleep(tries)
            except Exception as ex:
                log("An unexpected exception was caught:", ex)

def get_next_current_id(_json_data):
    items = _json_data['items']
    return items[-1]['id']


# initialize env and load config
log("Reading configuration")
config = {
    **dotenv.dotenv_values(".env"),
    **os.environ
}

log("Checking if configuration satisfies minimal config")
if not can_run():
    log("It's required to have at least " + ', '.join(required_config_keys) + " set in your env to run this program")
    exit(1)

log("Setting up cookies")
cookie_jar = setup_cookie_jar()

log("Preparing DB collections")
pr0_items_collection = get_collection("pr0items")

log("Determine starting position")
current_id = determine_start()
log("About to read from id", current_id)

# start reading from remote
cancel = False

while not cancel:
    log("fetching data from remote starting with id", current_id)
    json_data = fetch_json_data(
        "https://pr0gramm.com/api/items/get?older=" + str(current_id) + "&flags=" + str(content_flags))
    process_media_metadata(json_data)
    download_medias(json_data)
    current_id = get_next_current_id(json_data)
    if current_id <= 1:
        cancel = True
