import http.cookiejar
import json
import os
import urllib.request
import time
from datetime import datetime

import dotenv
from pymongo import MongoClient


def get_collection():
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    mydb = client["pr0loader"]
    mycol = mydb["pr0Item"]
    return mycol


def concat(message):
    result = ''
    for item in message:
        result = result + " " + str(item)
    return result


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
        max_value = collection.find(filter=_filter, projection=project, sort=sort, limit=limit)[0]['id']
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
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    result = opener.open("https://pr0gramm.com/api/items/get&flags=" + str(15))
    data = result.read()
    encoding = result.info().get_content_charset('utf-8')
    json_data = json.loads(data.decode(encoding))
    opener.close()
    return json_data['items'][0]['id']


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


# initialize env and load config
log("Reading configuration")
config = {
    **dotenv.dotenv_values(".env"),
    **os.environ
}
required_config_keys = ['ME', 'CONSENT', 'MONGODB_STRING']
can_run = True
for key in required_config_keys:
    if key not in config:
        can_run = False

if not can_run:
    log("It's required to have at least " + ', '.join(required_config_keys) + " set in your env to run this program")
    exit(1)

# create cookie jar and load data
cookie_jar = http.cookiejar.CookieJar()
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

cookie_jar.set_cookie(me_cookie)
cookie_jar.set_cookie(consent_cookie)

# Initialize database
collection = get_collection()

####


current_id = determine_start()
log("About to read from id", current_id)

# read filter settings (nsfl, nsfw, nsfp)
content_flags = 15  # this needs to be calculated late

# start reading from remote
cancel = False


def fetch_info_for_item(_id):
    log("Fetching info for id", _id)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    result = opener.open("https://pr0gramm.com/api/items/info?itemId=" + str(current_id) + "&flags=" + str(content_flags))
    data = result.read()
    encoding = result.info().get_content_charset('utf-8')
    json_data = json.loads(data.decode(encoding))
    opener.close()
    return json_data

while not cancel:
    log("fetching data from remote starting with id", current_id)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    result = opener.open("https://pr0gramm.com/api/items/get?older=" + str(current_id) + "&flags=" + str(content_flags))
    data = result.read()
    encoding = result.info().get_content_charset('utf-8')
    json_data = json.loads(data.decode(encoding))
    opener.close()
    # print(json.dumps(json_data, indent=4, sort_keys=True))
    for item in json_data['items']:
        add_info = fetch_info_for_item(item['id'])
        # print(item)
        item['comments'] = add_info['comments']
        item['tags'] = add_info ['tags']
        result = collection.insert_one(item)
        # print(result)
        current_id = item['id']
        log("Written item", current_id, "to database")
        # cancel = True
    if current_id == 1:
        cancel = True
    time.sleep(1)
    cancel = True
