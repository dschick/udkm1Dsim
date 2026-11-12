#!/usr/bin/env python
import numpy as np
import pytest

from udkm1Dsim import PhononNum, Result, u


@pytest.fixture
def result():
    result = Result("test", {"hash": "abc", "parameters": {"force recalc": False, "size": 3}})
    result.add_coord("delays", np.r_[0:3] * u.ps, "s")
    result.add_coord("distances", np.r_[0:4], "nm")
    result.add("temp_map", np.ones((3, 4)), ("delays", "distances"), "K")
    result.add("field", np.ones((3, 4, 2)) * (1 + 2j), ("delays", "distances"))
    return result


def test_access(result):
    assert result["temp_map"] is result.temp_map
    np.testing.assert_allclose(result.delays, [0, 1e-12, 2e-12])
    assert list(result.keys()) == ["temp_map", "field"]
    assert list(result.data) == ["temp_map", "field"]
    assert result.data["temp_map"] is result.temp_map
    assert list(result.coords) == ["delays", "distances"]
    assert dict(result).keys() == {"temp_map", "field"}
    assert "delays" in result and "temp_map" in result and "other" not in result
    assert len(result) == 2
    assert result.get("other") is None
    assert result.dims("temp_map") == ("delays", "distances")
    assert result.dims("delays") == ("delays",)
    assert result.unit("temp_map") == "K"
    assert result.quantity("temp_map").units == u.K
    assert result.quantity("field").dimensionless
    np.testing.assert_allclose(result.quantity("delays", "ps").magnitude, [0, 1, 2])
    assert "temp_map" in dir(result)
    with pytest.raises(KeyError, match="other is not part of the result"):
        result["other"]
    with pytest.raises(AttributeError, match="other is not part of the result"):
        result.other


def test_aliases(result, tmp_path):
    result.add("reflectivity", np.ones((3, 4)), ("delays", "distances"), aliases="R")
    result.add("reflectivity_phi", np.zeros((3, 4)), ("delays", "distances"), aliases=("R_phi",))
    assert result.R is result.reflectivity
    assert result["R_phi"] is result.reflectivity_phi
    assert "R" in result and "R" not in result.keys() and "R" not in result.data
    assert result.aliases == {"R": "reflectivity", "R_phi": "reflectivity_phi"}
    assert result.aliases_of("R") == ("R",)
    assert result.dims("R") == ("delays", "distances")
    assert result.quantity("R").dimensionless
    assert "R" in dir(result)
    assert "R_phi" in repr(result)
    selected = result.select("R")
    assert list(selected.keys()) == ["reflectivity"] and selected.aliases == {"R": "reflectivity"}
    with pytest.raises(KeyError, match="already a name"):
        result.add("other", np.ones(3), aliases="temp_map")
    with pytest.raises(KeyError, match="already an alias"):
        result.add("other", np.ones(3), aliases="R")
    with pytest.raises(KeyError, match="already an alias"):
        result.add("R", np.ones(3))
    with pytest.raises(ValueError, match="reserved"):
        result.add("other", np.ones(3), aliases="data")
    pytest.importorskip("h5py")
    result.save(tmp_path / "aliases.h5")
    assert Result.load(tmp_path / "aliases.h5").aliases == result.aliases


def test_no_unpacking(result):
    with pytest.raises(TypeError, match="cannot be iterated or unpacked"):
        temp_map, field = result
    with pytest.raises(TypeError, match="cannot be iterated or unpacked"):
        list(result)


def test_add_checks_dims(result):
    with pytest.raises(ValueError, match="does not agree"):
        result.add("wrong", np.ones((4, 3)), ("delays", "distances"))
    with pytest.raises(ValueError, match="dims are given"):
        result.add("wrong", np.ones(3), ("delays", "distances"))
    with pytest.raises(KeyError, match="coordinate"):
        result.add("delays", np.ones(3))
    with pytest.raises(KeyError, match="data entry"):
        result.add_coord("temp_map", np.ones(3))
    for name in ("data", "coords", "attrs", "save", "keys"):
        with pytest.raises(ValueError, match="reserved"):
            result.add(name, np.ones(3))
    with pytest.raises(ValueError, match="identifier"):
        result.add_coord("2nd axis", np.ones(3))


def test_select(result):
    selected = result.select("field")
    assert list(selected.keys()) == ["field"]
    assert "delays" in selected.coords
    assert selected.attrs["hash"] == "abc"


def test_repr(result):
    assert repr(result).startswith("Result: test")
    assert "temp_map" in repr(result)
    assert result._repr_html_().startswith("<h3>Result: test</h3>")


def test_save_load(result, tmp_path):
    h5py = pytest.importorskip("h5py")
    result.add("list", [np.ones(2), np.ones(3)])
    filename = tmp_path / "result.h5"
    with pytest.warns(UserWarning, match="list is no array"):
        result.save(filename)
    loaded = Result.load(filename)
    assert list(loaded.keys()) == ["temp_map", "field"]
    np.testing.assert_array_equal(loaded.field, result.field)
    np.testing.assert_array_equal(loaded.delays, result.delays)
    assert loaded.unit("distances") == "nm"
    assert loaded.dims("field") == ("delays", "distances")
    assert loaded.attrs["title"] == "test"
    assert loaded.attrs["parameters"] == {"force recalc": False, "size": 3}
    # the 1D coordinates are dimension scales of the data
    with h5py.File(filename, "r") as file:
        assert file["data/temp_map"].dims[0].label == "delays"
        assert file["data/temp_map"].dims[1][0].name == "/coords/distances"


def test_to_xarray(result):
    pytest.importorskip("xarray")
    dataset = result.to_xarray()
    assert dataset["temp_map"].dims == ("delays", "distances")
    assert dataset["field"].dims == ("delays", "distances", "field_dim_2")
    assert dataset["temp_map"].attrs["units"] == "K"
    np.testing.assert_allclose(dataset["delays"], result.delays)
    assert dataset.attrs["hash"] == "abc"


def test_chained_results(structure, tmp_path):
    """an upstream result is accepted instead of the arrays, its coordinates
    are passed on, and the result is cached in a HDF5 file"""
    delays = np.r_[-1:5:0.1] * u.ps
    N = structure.get_number_of_layers()
    temp_map = 300 * np.ones((len(delays), N))
    temp_map[delays.magnitude >= 0, :5] += 100
    delta_temp_map = np.zeros_like(temp_map)
    delta_temp_map[1:] = np.diff(temp_map, axis=0)
    temp = Result("temperature map")
    temp.add_coord("delays", delays, "s")
    temp.add("temp_map", temp_map, ("delays", "distances"), "K")
    temp.add("delta_temp_map", delta_temp_map, ("delays", "distances"), "K")
    pnum = PhononNum(structure, True, cache_dir=str(tmp_path), progress_bar=False)
    strain = pnum.get_strain_map(delays, temp)
    reference = pnum.get_strain_map(delays, temp_map, delta_temp_map)
    np.testing.assert_array_equal(strain.strain_map, reference.strain_map)
    assert strain.dims("strain_map") == ("delays", "distances")
    np.testing.assert_allclose(strain.distances, structure.get_distances_of_layers(False)[2])
    assert strain.attrs["simulation"] == "PhononNum"
    assert strain.attrs["hash"] == pnum.get_hash(delays, temp_map, delta_temp_map)
    assert "structure mixed" in strain.attrs["structure"]
    assert any(path.name.startswith("strain_map_num_") for path in tmp_path.iterdir())
    pnum.force_recalc = False
    np.testing.assert_array_equal(
        pnum.get_strain_map(delays, temp).strain_map, reference.strain_map
    )
    with pytest.raises(ValueError, match="delta_temp_map must be given"):
        pnum.get_strain_map(delays, temp_map)
