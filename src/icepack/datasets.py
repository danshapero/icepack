# Copyright (C) 2019-2025 by Daniel Shapero <shapero@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Routines for fetching the glaciological data sets used in the demos"""

import os
import gzip
import pathlib
import shutil
from getpass import getpass
import requests
import pooch
import earthaccess


def _fetch_nsidc(destination=None, **kwargs):
    earthaccess.login()
    results = earthaccess.search_data(**kwargs)
    if not results:
        raise ValueError(f"No results from Earthdata search for `{kwargs}`!")
    destination = destination or pooch.os_cache("icepack")
    return earthaccess.download(results, destination)


def _fetch_nsidc_v0(search_keys, filename, extra, destination):
    earthaccess.login()
    eaauth = earthaccess.__auth__
    auth = (eaauth.username, eaauth.password)

    # The URL to download the data on NSIDC isn't the real one, when you do a
    # `get` request it then redirects you to the real URL.
    results = earthaccess.search_datasets(**search_keys)
    assert len(results) == 1
    initial_urls = results[0].summary()["get-data"]
    assert len(initial_urls) == 1
    initial_url = initial_urls[0] + extra + filename
    real_url = requests.get(initial_url).url

    # TODO: progress bar
    download_path = destination / pathlib.Path(filename)
    with requests.get(real_url, auth=auth, stream=True) as request:
        request.raise_for_status()
        with open(download_path, "wb") as output_file:
            for chunk in request.iter_content(chunk_size=8192):
                output_file.write(chunk)

    return str(download_path)


def fetch_measures_antarctica(destination=None):
    r"""Fetch the MEaSUREs Antarctic velocity map"""
    return _fetch_nsidc(destination, short_name="NSIDC-0754")


def fetch_measures_greenland(destination=None):
    r"""Fetch the MEaSUREs Greenland velocity map"""
    criteria = {"granule_name": "greenland_vel_mosaic200_2015_2016*"}
    return _fetch_nsidc(destination, short_name="NSIDC-0478", **criteria)


def fetch_bedmachine_antarctica(destination=None):
    r"""Fetch the BedMachine map of Antarctic ice thickness, surface elevation,
    and bed elevation"""
    return _fetch_nsidc(destination, short_name="NSIDC-0756")


def fetch_bedmachine_greenland(destination=None):
    r"""Fetch the BedMachine map of Greenland ice thickness, surface elevation,
    and bed elevation"""
    criteria = {"version": "6", "granule_name": "*v*.nc"}
    return _fetch_nsidc(destination, short_name="IDBMG4", **criteria)


def fetch_mosaic_of_antarctica(destination=None):
    r"""Fetch the MODIS optical image mosaic of Antarctica"""
    destination = pathlib.Path(destination or pooch.os_cache("icepack"))
    search = {"short_name": "NSIDC-0593", "version": "2"}
    filename = "moa750_2009_hp1_v02.0.tif"
    extra = "geotiff/"
    gz_filename = _fetch_nsidc_v0(search, filename + ".gz", extra, destination)

    # Unzip the image
    result_path = destination / pathlib.Path(filename)
    with gzip.open(gz_filename, "rb") as gz_file:
        with open(result_path, "wb") as output_file:
            shutil.copyfileobj(gz_file, output_file)

    return str(result_path)


def fetch_mosaic_of_greenland(destination=None):
    r"""Fetch the MODIS optical image mosaic of Greenland"""
    criteria = {"granule_name": "mog100_2015_hp1*.tif"}
    return _fetch_nsidc(destination, short_name="NSIDC-0547", **criteria)


def fetch_randolph_glacier_inventory(destination=None):
    r"""Fetch the Alaska segment of the Randolph Glacier Inventory"""
    destination = pathlib.Path(destination or pooch.os_cache("icepack"))
    search = {"short_name": "NSIDC-0770", "version": "7"}
    filename = f"RGI2000-v7.0-G-01_alaska.zip"
    extra = "regional_files/RGI2000-v7.0-G/"
    zip_filename = _fetch_nsidc_v0(search, filename, extra, destination)
    return zip_filename


_outlines_url = "https://raw.githubusercontent.com/icepack/glacier-meshes"
_outlines_commit = "5906b7c21d844a982aa012e934fe29b31ef13d41"
outlines = pooch.create(
    path=pooch.os_cache("icepack"),
    base_url=f"{_outlines_url}/{_outlines_commit}/glaciers/",
    registry={
        "amery.geojson": "md5:b9a32abaacc3a36d5b696a26c2bd1b9b",
        "filchner-ronne.geojson": "md5:7876e9fad2fe74a99f3b1ff92e12dc3c",
        "getz.geojson": "md5:31dc3f10c0a06c05020683e8cb5a9f59",
        "helheim.geojson": "md5:21b754c088ceeb5995295a6ce54783e0",
        "hiawatha.geojson": "md5:3b0aa71d21641792b1bbbda35e185cca",
        "jakobshavn.geojson": "md5:baf707914993fb052e00024ccdceab92",
        "larsen-2015.geojson": "md5:317ba73b8a2370ec0832b0bc0bcfc986",
        "larsen-2018.geojson": "md5:cccb22fd94143d6ccbb4aaa08dee6cad",
        "larsen-2019.geojson": "md5:3188635279f93e863ae800fecb9d085a",
        "pine-island.geojson": "md5:2ebfb7a321568dcd481771ab3f0993c6",
        "ross.geojson": "md5:a4cf6461607c90961280e5afbab1123b",
    },
)


def get_glacier_names():
    r"""Return the names of the glaciers for which we have outlines that you
    can fetch"""
    return [
        os.path.splitext(os.path.basename(filename))[0]
        for filename in outlines.registry.keys()
    ]


def fetch_outline(name):
    r"""Fetch the outline of a glacier as a GeoJSON file"""
    names = get_glacier_names()
    if name not in names:
        raise ValueError("Glacier name '%s' not in %s" % (name, names))
    downloader = pooch.HTTPDownloader(progressbar=True)
    return outlines.fetch(name + ".geojson", downloader=downloader)
