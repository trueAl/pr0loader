# pr0loader
## Purpose
Create a local copy of the pr0gramms data (images, videos) and all tags and comments to train models downstream.

Current first intended use is to create the pr0tag0nist, but other users might find other use cases for the resulting
datasets.


## Usage
### Setting up .env
Copy your `template.env` over to `.env` and modify the values accordingly to your needs.

Grab the cookies from you dev tools, or implement a shiny solution here :-)

### 01_pr0loader.py
This script can be configured through the `.env` as well. The full `CONTENT_FLAGS` were 15 at the time of writing, and
can be obtained from the API calls in your browser.

If you run a `FULL_UPDATE = True`, expect roughly `8TB` of data, if you run for all flags.

### 02_prepare_csv.py
This script fetches the data from the mongodb, verifies the files exist in filesystem and creates a csv.

Why is the data in the mongo, and not directly written to CSV in 01? Because this way, we can tweak the mongodb
queries to select the data and store data which is irrelevant for certain tasks, like RESNET training does not require
all the comments, but they might come handy thinking of a pr0 LLM ;)

### 03_prepare_dataset.py
This scripts job is to load the CSV from 02 into a custom Dataset class, and pre-render all images to n*3*224*224 and store the result in a HDF5 file.

I selected this approach, so I would be able to do the resize/crop ops for roughly 1.2M images at time of writing only once, instead of doing it several times.
