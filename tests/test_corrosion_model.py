from corrosion_model import (
    ZINC_B1,
    iso_9223_steel_rcorr,
    iso_9223_zinc_rcorr,
    iso_9224_zinc_loss_um,
    zinc_corrosivity_category,
)


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol


def test_known_zinc_rcorr_value():
    # Regression value from the implemented ISO 9223 Eq. 2 form.
    r = float(iso_9223_zinc_rcorr(T_C=15, RH_pct=70, Pd_mg_m2_d=4, Sd_mg_m2_d=3))
    approx(r, 0.6217112857992437, 1e-12)


def test_zinc_category_boundaries():
    assert zinc_corrosivity_category(0.1) == "C1"
    assert zinc_corrosivity_category(0.1001) == "C2"
    assert zinc_corrosivity_category(2.1) == "C3"
    assert zinc_corrosivity_category(9.0) == "CX"
    assert zinc_corrosivity_category(30.0) == ">CX"


def test_steel_rcorr_is_positive_for_typical_inputs():
    r = iso_9223_steel_rcorr(20.0, 70.0, 5.0, 10.0)

    assert r > 0.0


def test_9224_30_year_linear_tail():
    d = float(iso_9224_zinc_loss_um(1.0, 30, b=ZINC_B1, after_20="linear"))
    # 20^0.813 + 0.813 * 20^(0.813-1) * 10
    approx(d, 16.06486657718427, 1e-12)
