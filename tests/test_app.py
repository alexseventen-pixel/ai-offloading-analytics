"""Smoke + unit tests. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis  # noqa: E402
import app as A  # noqa: E402
import db  # noqa: E402


class DeriveTests(unittest.TestCase):
    def test_offloading_reverse_coding(self):
        r = db.derive({"q4_ai_frequency": 1, "q8_ask_ai_first": 1, "q9_copy_directly": 1,
                       "q10_rewrite_own_words": 5, "q11_check_correctness": 5, "q12_could_do_without_ai": 5})
        self.assertEqual(r["offloading_score"], 1.0)
        r = db.derive({"q4_ai_frequency": 5, "q8_ask_ai_first": 5, "q9_copy_directly": 5,
                       "q10_rewrite_own_words": 1, "q11_check_correctness": 1, "q12_could_do_without_ai": 1})
        self.assertEqual(r["offloading_score"], 5.0)

    def test_missing_items_tolerated(self):
        r = db.derive({"q4_ai_frequency": 2, "q8_ask_ai_first": 2, "q9_copy_directly": 2, "q10_rewrite_own_words": 4})
        self.assertEqual(r["offloading_score"], 2.0)
        self.assertIsNone(db.derive({"q4_ai_frequency": 2})["offloading_score"])

    def test_ct_and_flags(self):
        r = db.derive({"ct_detection": 2, "ct_justification": 1, "luria_trial1": 3, "luria_delayed": 2,
                       "schulte_seconds": 70, "digit_span_total": 6})
        self.assertEqual(r["ct_total"], 3)
        self.assertEqual(r["low_ct"], 1)
        self.assertEqual(r["low_attention"], 1)

    def test_critical_r_matches_thesis(self):
        self.assertAlmostEqual(analysis.critical_r(12), 0.576, places=3)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.c = A.app.test_client()

    def test_pages(self):
        for u in ["/", "/dashboard", "/dashboard?grade=9", "/records", "/records/P1001", "/predict",
                  "/predict?demo=high", "/model", "/data", "/export.csv", "/api/records"]:
            self.assertEqual(self.c.get(u).status_code, 200, u)

    def test_crud_cycle(self):
        n0 = len(db.load_df())
        r = self.c.post("/records/new", data={"grade": "8", "q4_ai_frequency": "2", "ct_detection": "3",
                                              "ct_justification": "2"})
        sid = r.headers["Location"].rsplit("/", 1)[1]
        self.assertEqual(db.get(sid)["ct_total"], 5)
        self.c.post(f"/records/{sid}/edit", data={"grade": "9", "ct_detection": "4", "ct_justification": "4"})
        self.assertEqual(db.get(sid)["ct_total"], 8)
        self.c.post(f"/records/{sid}/delete")
        self.assertIsNone(db.get(sid))
        self.assertEqual(len(db.load_df()), n0)

    def test_validation(self):
        r = self.c.post("/records/new", data={"grade": "8", "luria_trial1": "15"})
        self.assertIn(b"between", r.data)

    def test_api_predict(self):
        out = self.c.post("/api/predict", json={"grade": 9, "q4_ai_frequency": 5, "q8_ask_ai_first": 5,
                                                "q9_copy_directly": 4, "q10_rewrite_own_words": 1,
                                                "q11_check_correctness": 1, "q12_could_do_without_ai": 1}).json
        self.assertGreater(out["low_ct"]["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
