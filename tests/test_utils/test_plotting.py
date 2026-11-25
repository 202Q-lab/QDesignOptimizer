from qdesignoptimizer.utils.plotting import load_data_by_date


def test_load_data_by_date_includes_both_bounds_and_filters_experiment(tmp_path):
    names = [
        "chip_20250101-000000.npy",
        "chip_20250102-120000.npy",
        "chip_20250103-235959.npy",
        "chip_20250104-000000.npy",
        "other_20250102-120000.npy",
        "chip_20250102-120000_design_variables.json",
    ]
    for name in names:
        (tmp_path / name).write_text("")

    files = load_data_by_date(
        str(tmp_path), "chip", "20250102-120000", "chip_20250103-235959"
    )
    assert sorted(str(f).rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for f in files) == [
        "chip_20250102-120000.npy",
        "chip_20250103-235959.npy",
    ]
