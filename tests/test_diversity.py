from scripts.quality.diversity import (
    distinct_n,
    vocabulary_size,
    label_distribution,
    normalized_entropy,
    diversity_report,
)

text_fn = lambda r: r.get("body", "")


def test_distinct_n_repeated_text_is_low():
    repetitive = ["the cat sat", "the cat sat", "the cat sat"]
    varied = ["the cat sat", "a dog ran", "birds fly south"]
    assert distinct_n(varied, 2) > distinct_n(repetitive, 2)


def test_vocabulary_size_counts_unique_tokens():
    assert vocabulary_size(["hello world", "hello there"]) == 3


def test_label_distribution_counts(valid_records):
    dist = label_distribution(valid_records, "sentiment")
    assert dist["negative"] == 2
    assert dist["positive"] == 1


def test_normalized_entropy_balanced_is_high():
    balanced = {"a": 5, "b": 5}
    skewed = {"a": 9, "b": 1}
    assert normalized_entropy(balanced) == 1.0
    assert normalized_entropy(skewed) < 1.0


def test_normalized_entropy_single_class_is_zero():
    assert normalized_entropy({"a": 10}) == 0.0


def test_diversity_report_shape(valid_records):
    report = diversity_report(valid_records, text_fn, ["category", "sentiment"], ngram_n=2)
    assert report["n_records"] == 3
    assert "distinct_2" in report
    assert "category" in report["labels"]
    assert "balance" in report["labels"]["category"]
