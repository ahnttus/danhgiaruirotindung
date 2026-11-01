import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag_service import classify_history_query


class QueryRouterTests(unittest.TestCase):
    def test_single_record_id(self):
        self.assertEqual(classify_history_query("Phân tích hồ sơ số 125")['record_ids'], [125])

    def test_compare_two_records(self):
        plan = classify_history_query("So sánh hồ sơ 125 với hồ sơ 128")
        self.assertEqual(plan["intent"], "by_ids")
        self.assertEqual(plan["record_ids"], [125, 128])

    def test_current_record(self):
        self.assertEqual(classify_history_query("Vì sao hồ sơ hiện tại có PD cao?")["intent"], "current")

    def test_grade_filter(self):
        plan = classify_history_query("Tìm các hồ sơ hạng D")
        self.assertEqual(plan, {"intent": "by_grade", "grade": "D"})

    def test_statistics(self):
        self.assertEqual(classify_history_query("Có bao nhiêu hồ sơ đã lưu?")["intent"], "stats")

    def test_grade_count(self):
        plan = classify_history_query("Có bao nhiêu hồ sơ hạng D?")
        self.assertEqual(plan, {"intent": "grade_count", "grade": "D"})

    def test_general_concept_is_not_record_search(self):
        self.assertEqual(classify_history_query("Hồ sơ như thế nào thì được đánh giá là tốt?")["intent"], "none")

    def test_missing_reference_requires_clarification(self):
        self.assertEqual(classify_history_query("Tìm hồ sơ đã được ghi trước đó")["intent"], "clarify")


if __name__ == "__main__":
    unittest.main()
