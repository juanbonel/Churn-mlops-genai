from churn_project.data import FEATURES, TARGET, make_churn_dataset


def test_synthetic_data_is_reproducible_and_has_expected_schema():
    first = make_churn_dataset(n_samples=100, random_state=7)
    second = make_churn_dataset(n_samples=100, random_state=7)
    assert first.equals(second)
    assert set(FEATURES + [TARGET]).issubset(first.columns)
    assert first[TARGET].isin([0, 1]).all()


def test_shift_changes_feature_distributions():
    reference = make_churn_dataset(n_samples=1000, random_state=42)
    shifted = make_churn_dataset(n_samples=1000, random_state=42, shift=True)
    assert shifted["contract_type"].value_counts(normalize=True).get("monthly", 0) > (
        reference["contract_type"].value_counts(normalize=True).get("monthly", 0)
    )
