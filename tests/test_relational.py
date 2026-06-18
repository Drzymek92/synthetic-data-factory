import json

import pytest

import scripts.llm_client as llm_client
from scripts.config import GenConfig, QualityConfig
from scripts.domain import load_domain
from scripts.export import export_relational
from scripts.relational import (
    generation_order,
    generate_relational,
    check_referential_integrity,
    validate_entities,
    evaluate_entities,
    export_duckdb,
)

# A superset of every entity's CONTENT fields. Pydantic ignores extras, so the same
# record validates for sellers/buyers/offers/orders/order_items — the generator keeps
# whatever each entity's schema needs and drops the rest.
SUPERSET = {
    "name": "Test Shop Listing",
    "rating": 4,
    "login": "user_01",
    "smart": "yes",
    "locale": "pl",
    "category": "electronics",
    "price": 100,
    "status": "DELIVERED",
    "delivery_method": "COURIER",
    "quantity": 2,
}


@pytest.fixture
def marketplace_spec():
    return load_domain("marketplace")


@pytest.fixture
def fake_llm(monkeypatch):
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        return {"records": [dict(SUPERSET) for _ in range(10)]}

    monkeypatch.setattr(llm_client, "llm_json", fake)


def test_generation_order_parents_before_children(marketplace_spec):
    order = generation_order(marketplace_spec)
    pos = {name: i for i, name in enumerate(order)}
    # offers ref sellers; orders ref buyers+sellers; order_items per_parent orders + ref offers
    assert pos["sellers"] < pos["offers"]
    assert pos["buyers"] < pos["orders"]
    assert pos["sellers"] < pos["orders"]
    assert pos["orders"] < pos["order_items"]
    assert pos["offers"] < pos["order_items"]


def test_generate_relational_assigns_ids_and_fks(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=1), marketplace_spec)
    assert set(tables) == set(marketplace_spec["entities"])
    for ename, rows in tables.items():
        assert rows, f"{ename} produced no rows"
        assert all("id" in r for r in rows)
    # FK columns are present on child entities
    assert all("seller_id" in o for o in tables["offers"])
    assert all({"buyer_id", "seller_id"} <= set(o) for o in tables["orders"])


def test_referential_integrity_holds(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=2), marketplace_spec)
    assert check_referential_integrity(tables, marketplace_spec) == []


def test_per_parent_cardinality_and_parent_link(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=3), marketplace_spec)
    n_orders = len(tables["orders"])
    n_items = len(tables["order_items"])
    assert n_orders <= n_items <= n_orders * 3  # min 1, max 3 per order
    order_ids = {o["id"] for o in tables["orders"]}
    assert all(item["order_id"] in order_ids for item in tables["order_items"])


def test_validate_entities_all_valid(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=4), marketplace_spec)
    results = validate_entities(tables, marketplace_spec)
    assert all(r["errors"] == 0 and r["valid"] > 0 for r in results.values())


def test_unknown_ref_raises(tmp_path):
    (tmp_path / "broken.yaml").write_text(
        "name: broken\n"
        "entities:\n"
        "  child:\n"
        "    fields:\n"
        "      x: {type: str}\n"
        "    relationships:\n"
        "      p_id: {ref: ghost}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_domain("broken", domains_dir=tmp_path)


# --- C: cross-entity coherence (copy + constrained FK) ---

def test_copy_denormalizes_offer_price(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=6), marketplace_spec)
    offer_price = {o["id"]: o["price"] for o in tables["offers"]}
    assert tables["order_items"], "no line items generated"
    for item in tables["order_items"]:
        assert "unit_price" in item
        assert item["unit_price"] == offer_price[item["offer_id"]]


def test_constrained_fk_keeps_offer_with_order_seller(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=7), marketplace_spec)
    order_seller = {o["id"]: o["seller_id"] for o in tables["orders"]}
    offer_seller = {o["id"]: o["seller_id"] for o in tables["offers"]}
    sellers_with_offers = set(offer_seller.values())
    for item in tables["order_items"]:
        os_ = order_seller[item["order_id"]]
        # Coherence is guaranteed only when the order's seller actually has an offer;
        # otherwise the generator falls back to an unconstrained pick (logged).
        if os_ in sellers_with_offers:
            assert offer_seller[item["offer_id"]] == os_


def test_assign_fks_match_filters_candidates():
    from scripts.relational import _assign_fks
    import random
    spec = {
        "entities": {
            "orders": {}, "offers": {}, "order_items": {
                "relationships": {
                    "order_id": {"ref": "orders"},
                    "offer_id": {"ref": "offers", "match": {"field": "seller_id", "via": "order_id"}},
                },
            },
        }
    }
    tables = {
        "orders": [{"id": "ORD-1", "seller_id": "SEL-A"}],
        "offers": [
            {"id": "OFR-1", "seller_id": "SEL-A"},
            {"id": "OFR-2", "seller_id": "SEL-B"},
            {"id": "OFR-3", "seller_id": "SEL-B"},
        ],
    }
    espec = spec["entities"]["order_items"]
    for _ in range(20):
        rec: dict = {}
        _assign_fks(rec, "order_items", espec, tables, spec, random.Random(0),
                    fixed={"order_id": "ORD-1"})
        assert rec["offer_id"] == "OFR-1"  # only SEL-A offer is eligible


def test_apply_copies_pulls_referenced_field():
    from scripts.relational import _apply_copies
    spec = {"entities": {"offers": {}, "order_items": {
        "relationships": {"offer_id": {"ref": "offers"}},
        "copy": {"unit_price": {"from": "offer_id", "field": "price"}},
    }}}
    tables = {"offers": [{"id": "OFR-1", "price": 399}]}
    rec = {"offer_id": "OFR-1"}
    _apply_copies(rec, spec["entities"]["order_items"], tables, spec)
    assert rec["unit_price"] == 399


# --- A: per-entity quality harness (flag-only dedup, diversity, judge) ---

def test_evaluate_entities_dedup_is_flag_only(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=10), marketplace_spec)
    before = {k: len(v) for k, v in tables.items()}
    q = evaluate_entities(tables, marketplace_spec, QualityConfig(run_judge=False))
    after = {k: len(v) for k, v in tables.items()}
    assert before == after                              # nothing removed
    # fake LLM gives every offer the same name → all pairs are near-duplicates, flagged
    assert q["offers"]["near_duplicate_pairs"] > 0
    # orders declare label_fields but no text_field → label balance only
    assert "labels" in q["orders"]


def test_evaluate_entities_judge_offline(monkeypatch, marketplace_spec):
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        if prompt.startswith("Score this record"):
            return {"realism": 4, "category_fit": 5, "issue": ""}
        return {"records": [dict(SUPERSET) for _ in range(10)]}

    monkeypatch.setattr(llm_client, "llm_json", fake)
    tables = generate_relational(GenConfig(seed=11), marketplace_spec)
    q = evaluate_entities(tables, marketplace_spec, QualityConfig(run_judge=True))
    assert q["offers"]["judge"]["judged"] == len(tables["offers"])
    assert q["offers"]["judge"]["overall_mean"] == 4.5


def test_evaluate_entities_respects_no_judge(fake_llm, marketplace_spec):
    tables = generate_relational(GenConfig(seed=12), marketplace_spec)
    q = evaluate_entities(tables, marketplace_spec, QualityConfig(run_judge=False))
    assert "judge" not in q.get("offers", {})           # judge skipped when run_judge=False


# --- B: deterministic (non-LLM) fill for structural entities ---

def test_synthetic_fill_makes_no_llm_calls(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called for a fill:random entity")

    monkeypatch.setattr(llm_client, "llm_json", boom)
    spec = {
        "name": "structural",
        "entities": {
            "widgets": {
                "id_prefix": "W", "count": 5, "fill": "random",
                "fields": {
                    "size": {"type": "int", "ge": 1, "le": 4},
                    "color": {"type": "enum", "values": ["red", "green", "blue"]},
                },
            },
        },
    }
    tables = generate_relational(GenConfig(seed=1), spec)
    assert len(tables["widgets"]) == 5
    for r in tables["widgets"]:
        assert 1 <= r["size"] <= 4
        assert r["color"] in {"red", "green", "blue"}
        assert "id" in r


def test_marketplace_order_items_filled_in_bounds(fake_llm, marketplace_spec):
    # order_items is fill:random; quantity must respect its ge/le without an LLM call.
    tables = generate_relational(GenConfig(seed=13), marketplace_spec)
    assert tables["order_items"]
    assert all(1 <= it["quantity"] <= 5 for it in tables["order_items"])


# --- Configurator: scale controls + per-field distributions + date type, synthetic default ---

def test_apply_scale_multiplies_and_overrides():
    from scripts.relational import apply_scale
    spec = {"entities": {
        "a": {"count": 10}, "b": {"count": 4},
        "kids": {"per_parent": {"parent": "a"}},   # no count — unaffected
    }}
    apply_scale(spec, scale=5, count_overrides={"b": 100})
    assert spec["entities"]["a"]["count"] == 50      # scaled
    assert spec["entities"]["b"]["count"] == 100     # override wins over scale
    assert "count" not in spec["entities"]["kids"]   # per_parent untouched


def test_default_content_mode_is_synthetic(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an entity with no `generate: llm` must not call the LLM")

    monkeypatch.setattr(llm_client, "llm_json", boom)
    spec = {"name": "d", "entities": {"things": {
        "id_prefix": "T", "count": 3,
        "fields": {"k": {"type": "enum", "values": ["x", "y"]}},
    }}}
    tables = generate_relational(GenConfig(seed=1), spec)
    assert len(tables["things"]) == 3


def test_synthetic_enum_weights_skew():
    from scripts.relational import _synthetic_fill
    import random
    espec = {"fields": {"s": {"type": "enum", "values": ["yes", "no"], "weights": {"yes": 3, "no": 7}}}}
    rows = _synthetic_fill(espec, 2000, random.Random(0))
    frac_yes = sum(r["s"] == "yes" for r in rows) / len(rows)
    assert 0.25 < frac_yes < 0.35          # ~0.3 by design


def test_synthetic_int_choices_and_bounds():
    from scripts.relational import _synthetic_fill
    import random
    espec = {"fields": {"q": {"type": "int", "ge": 1, "le": 5, "choices": [1, 1, 2, 3]}}}
    rows = _synthetic_fill(espec, 200, random.Random(0))
    assert {r["q"] for r in rows} <= {1, 2, 3}    # only sampled from choices


def test_synthetic_date_in_range():
    from scripts.relational import _synthetic_fill
    from datetime import date
    import random
    espec = {"fields": {"d": {"type": "date", "anchor": "2026-06-17",
                              "min_offset_days": -10, "max_offset_days": 0}}}
    rows = _synthetic_fill(espec, 100, random.Random(0))
    for r in rows:
        d = date.fromisoformat(r["d"])
        assert date(2026, 6, 7) <= d <= date(2026, 6, 17)


def test_export_duckdb_is_queryable_and_enforces_fks(fake_llm, marketplace_spec, tmp_path):
    import duckdb
    tables = generate_relational(GenConfig(seed=14), marketplace_spec)
    path = export_duckdb(tables, marketplace_spec, tmp_path, "20260618_000000")
    assert path.exists()

    con = duckdb.connect(str(path), read_only=True)
    try:
        # every entity is a table with the expected row count
        for ename, rows in tables.items():
            n = con.execute(f'SELECT count(*) FROM "{ename}"').fetchone()[0]
            assert n == len(rows)
        # a cross-table JOIN works out of the box (line item → offer → seller)
        joined = con.execute(
            'SELECT count(*) FROM order_items i '
            'JOIN offers o ON i.offer_id = o.id '
            'JOIN sellers s ON o.seller_id = s.id'
        ).fetchone()[0]
        assert joined == len(tables["order_items"])
    finally:
        con.close()

    # FK constraint is real: inserting an order with a non-existent buyer must fail
    con = duckdb.connect(str(path))
    try:
        import pytest as _pytest
        with _pytest.raises(Exception):
            con.execute(
                "INSERT INTO orders (id, status, delivery_method, placed_at, buyer_id, seller_id) "
                "VALUES ('ORD-BAD', 'NEW', 'COURIER', DATE '2026-01-01', 'BUY-DOESNOTEXIST', "
                "(SELECT id FROM sellers LIMIT 1))"
            )
    finally:
        con.close()


def test_export_relational_writes_files(fake_llm, marketplace_spec, tmp_path):
    tables = generate_relational(GenConfig(seed=5), marketplace_spec)
    written = export_relational(tables, marketplace_spec, tmp_path, "20260618_000000")
    csvs = [p for p in written if p.suffix == ".csv"]
    bundles = [p for p in written if p.suffix == ".json"]
    assert len(csvs) == len(marketplace_spec["entities"])
    assert len(bundles) == 1
    loaded = json.loads(bundles[0].read_text(encoding="utf-8"))
    assert set(loaded) == set(marketplace_spec["entities"])
