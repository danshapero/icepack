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

import firedrake
import geojson
import geopandas
import icepack
import numpy as np
from numpy import pi as π
import pytest


def needs_snapping():
    coords = [
        [(0.0, -1e-6), (1.0, 0.0), (1.0, 1.0 - 1e-6)],
        [(1.0, 1.0 + 1e-6), (0.0, 1.0), (0.0, 1e-6)],
    ]
    multi_line_string = geojson.MultiLineString(coords, validate=True)
    feature = geojson.Feature(geometry=multi_line_string, properties={})
    return geojson.FeatureCollection([feature])


def needs_reorienting():
    coords = [
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ]
    multi_line_string = geojson.MultiLineString(coords, validate=True)
    feature = geojson.Feature(geometry=multi_line_string, properties={})
    return geojson.FeatureCollection([feature])


def has_interior():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    outer_line_string = geojson.LineString(coords, validate=True)

    r = 1 / 8
    coords = [
        (0.5 + r * np.cos(θ), 0.5 + r * np.sin(θ)) for θ in np.linspace(0, 2 * π, 256)
    ]
    inner_line_string = geojson.LineString(coords, validate=True)

    outer_feature = geojson.Feature(geometry=outer_line_string, properties={})
    inner_feature = geojson.Feature(geometry=inner_line_string, properties={})
    return geojson.FeatureCollection([outer_feature, inner_feature])


test_data = [needs_snapping, needs_reorienting, has_interior]


@pytest.mark.parametrize("input_data", test_data)
def test_normalize(input_data):
    collection = input_data()
    result = icepack.meshing.normalize(collection)
    assert result == icepack.meshing.normalize(result)


@pytest.mark.skipif(not icepack.meshing._has_pygmsh, reason="No pygmsh")
@pytest.mark.parametrize("input_data", test_data)
def test_converting_to_geo(tmpdir, input_data):
    collection = input_data()
    geometry = icepack.meshing.collection_to_geo(collection, lcar=1e-2)
    assert geometry.get_code() is not None


@pytest.mark.parametrize("input_data", test_data)
def test_converting_to_gmsh(tmp_path, input_data):
    collection = input_data()
    geometry = icepack.meshing.collection_to_gmsh(collection, lcar=1e-2)
    filename = tmp_path / f"{input_data.__name__}.msh"
    geometry.write(filename)
    mesh = firedrake.Mesh(str(filename))
    assert mesh.num_cells() > 0


@pytest.mark.parametrize("input_data", test_data)
def test_converting_to_triangle(input_data):
    collection = input_data()
    lcar = 1e-1
    geometry = icepack.meshing.collection_to_triangle(
        collection, max_volume=0.5 * lcar**2
    )
    assert len(geometry.elements) > 0

    mesh = icepack.meshing.triangle_to_firedrake(geometry)
    assert mesh.num_vertices() > 0
    assert mesh.num_cells() > 0


def test_meshing_real_outlines(tmp_path):
    for glacier_name in icepack.datasets.get_glacier_names():
        outline_filename = icepack.datasets.fetch_outline(glacier_name)
        with open(outline_filename, "r") as outline_file:
            outline = geojson.load(outline_file)

        msh_filename = f"{tmp_path}/{glacier_name}.msh"
        icepack.meshing.collection_to_gmsh(outline).write(msh_filename)
        mesh = firedrake.Mesh(msh_filename)
        assert mesh.num_cells() > 0


@pytest.mark.skip(reason="EarthData auth required")
def test_meshing_rgi_polygon(tmp_path):
    rgi_filename = icepack.datasets.fetch_randolph_glacier_inventory("alaska")
    dataframe = geopandas.read_file(rgi_filename)
    entry = dataframe[dataframe["glac_name"] == "Gulkana Glacier"]
    outline_lat_lon = entry.geometry
    utm_crs = outline_lat_lon.estimate_utm_crs()
    outline_utm = outline_lat_lon.to_crs(utm_crs)
    outline_json = geojson.loads(outline_utm.to_json())
    outline = geojson.utils.map_tuples(lambda x: x[:2], outline_json)

    msh_filename = f"{tmp_path}/gulkana.msh"
    icepack.meshing.collection_to_msh(outline).write(msh_filename)
    mesh = firedrake.Mesh(msh_filename)
    assert mesh.num_cells() > 0
