from scripts.quality.dedup import find_near_duplicates, dedupe

text_fn = lambda r: r.get("body", "")


def test_find_near_duplicates_detects_identical_body(duplicate_records):
    pairs = find_near_duplicates(duplicate_records, text_fn, threshold=0.9)
    assert any(score >= 0.9 for _, _, score in pairs)


def test_dedupe_removes_one_of_a_pair(duplicate_records):
    kept, removed = dedupe(duplicate_records, text_fn, threshold=0.9)
    assert len(kept) == len(duplicate_records) - 1
    assert len(removed) == 1


def test_dedupe_keeps_distinct_records(valid_records):
    kept, removed = dedupe(valid_records, text_fn, threshold=0.9)
    assert len(kept) == len(valid_records)
    assert removed == []


def test_find_near_duplicates_handles_single_record(valid_records):
    assert find_near_duplicates(valid_records[:1], text_fn) == []
