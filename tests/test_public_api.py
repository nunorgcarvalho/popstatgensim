import importlib

import pytest

import popstatgensim as psg


def test_root_public_api_exposes_main_symbols():
    assert psg.Population.__name__ == "Population"
    assert psg.PopulationParams.__name__ == "PopulationParams"
    assert psg.SuperPopulation.__name__ == "SuperPopulation"
    assert psg.Trait.__name__ == "Trait"
    assert psg.GeneticEffect.__name__ == "GeneticEffect"
    assert psg.RandomEffect.__name__ == "RandomEffect"
    assert callable(psg.run_EO_AM)
    assert callable(psg.run_HEreg)
    assert callable(psg.run_REML)
    assert callable(psg.read_GRM_GCTA)
    assert callable(psg.read_table_GCTA)
    assert callable(psg.align_samples)
    assert callable(psg.subset_grm_by_ids)
    assert callable(psg.prepare_reml_inputs)
    assert not hasattr(psg, "run_HE_regression")


def test_root_public_api_exposes_main_namespaces():
    assert hasattr(psg, "genome")
    assert hasattr(psg, "traits")
    assert hasattr(psg, "pedigree")
    assert hasattr(psg, "simulation")
    assert hasattr(psg, "estimation")
    assert hasattr(psg, "plotting")
    assert hasattr(psg, "io")
    assert hasattr(psg, "utils")


@pytest.mark.parametrize(
    "legacy_module",
    [
        "popstatgensim.core_functions",
        "popstatgensim.export_functions",
        "popstatgensim.genetics",
        "popstatgensim.popgen_functions",
        "popstatgensim.popsim",
        "popstatgensim.relative_types",
        "popstatgensim.reml",
        "popstatgensim.statgen_functions",
    ],
)
def test_legacy_flat_modules_are_no_longer_importable(legacy_module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_module)
