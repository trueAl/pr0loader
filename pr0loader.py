import http.cookiejar
import json
import os
import urllib.request
from datetime import datetime

import dotenv
from pymongo import MongoClient


def get_database():
    connection_string = config['MONGODB_STRING']
    client = MongoClient(connection_string)
    return client['pr0loader']


def log(message: str):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("[", date_time, "] --- ", " " + message)


# initialize env and load config
log("Reading configuration")
config = {
    **dotenv.dotenv_values(".env"),
    **os.environ
}
required_config_keys = ['ME', 'CONSENT', 'LAST_IMAGE_ID', 'MONGODB_STRING']
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
consent_cookie = http.cookiejar.Cookie(version=0, name='consent', value=config['CONSENT'], port=None,
                                       port_specified=False,
                                       domain='pr0gramm.com', domain_specified=False, domain_initial_dot=False,
                                       path='/',
                                       path_specified=True, secure=True, expires=None, discard=True, comment=None,
                                       comment_url=None, rest={'HttpOnly': None}, rfc2109=False)

cookie_jar.set_cookie(me_cookie)
cookie_jar.set_cookie(consent_cookie)

# Initialize database
database = get_database()

# test = db["test"]
# item_1 = {
#     "_id": "U1IT00001",
#     "item_name": "Blender",
#     "max_discount": "10%",
#     "batch_number": "RR450020FRG",
#     "price": 340,
#     "category": "kitchen appliance"
# }
# test.insert_one(item_1)

# read start pointer from file (e.g. highest ID of image loaded
start_reading_from = config['LAST_IMAGE_ID']

# read filter settings (nsfl, nsfw, nsfp)
content_flags = 15  # this needs to be calculated late

# start reading from remote
current_id = start_reading_from
cancel = False
while not cancel:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    result = opener.open("https://pr0gramm.com/api/items/get?id=" + current_id + "&flags=" + str(content_flags))
    data = result.read()
    encoding = result.info().get_content_charset('utf-8')
    json_data = json.loads(data.decode(encoding))
    print(json_data)
    cancel = True
