import popstatgensim as psg


def test_root_public_api_exposes_main_symbols():
    assert psg.Population.__name__ == "Population"
    assert psg.SuperPopulation.__name__ == "SuperPopulation"
    assert psg.Trait.__name__ == "Trait"
    assert psg.GeneticEffect.__name__ == "GeneticEffect"
    assert callable(psg.run_HEreg)
    assert callable(psg.run_REML)


def test_root_public_api_exposes_main_namespaces():
    assert hasattr(psg, "genetics")
    assert hasattr(psg, "genome")
    assert hasattr(psg, "traits")
    assert hasattr(psg, "pedigree")
    assert hasattr(psg, "simulation")
    assert hasattr(psg, "estimation")
    assert hasattr(psg, "plotting")
    assert hasattr(psg, "io")
    assert hasattr(psg, "utils")
