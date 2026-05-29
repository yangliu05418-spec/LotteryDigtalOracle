import json
import unittest

from crawler_ssq import normalize_record, parse_payload, sort_records_ascending, strip_jsonp


class TestCrawlerSSQ(unittest.TestCase):
    def test_strip_jsonp_accepts_plain_json_and_jsonp(self):
        plain = '{"data": []}'
        wrapped = 'jQuery1122_123({"data": []});'
        self.assertEqual(strip_jsonp(plain), plain)
        self.assertEqual(strip_jsonp(wrapped), plain)

    def test_parse_payload_reads_plain_json_response(self):
        payload = parse_payload('{"pageNum":"1","pages":"1","total":"0","data":[]}')
        self.assertEqual(payload["pageNum"], "1")
        self.assertEqual(payload["data"], [])

    def test_normalize_record_extracts_csv_fields(self):
        raw = {
            "issue": "2026060",
            "openTime": "2026-05-28",
            "frontWinningNum": "07 09 10 16 22 27",
            "backWinningNum": "11",
        }
        self.assertEqual(
            normalize_record(raw),
            ["2026060", "2026-05-28", "07", "09", "10", "16", "22", "27", "11"],
        )

    def test_sort_records_ascending_by_issue_number(self):
        rows = [
            ["2026002", "2026-01-04", "01", "02", "03", "04", "05", "06", "07"],
            ["2003001", "2003-02-23", "10", "11", "12", "13", "26", "28", "11"],
            ["2026001", "2026-01-01", "01", "02", "03", "04", "05", "06", "07"],
        ]
        self.assertEqual([r[0] for r in sort_records_ascending(rows)], ["2003001", "2026001", "2026002"])

    def test_normalize_record_rejects_bad_ball_count(self):
        raw = {
            "issue": "2026060",
            "openTime": "2026-05-28",
            "frontWinningNum": "07 09 10",
            "backWinningNum": "11",
        }
        with self.assertRaises(ValueError):
            normalize_record(raw)


if __name__ == "__main__":
    unittest.main()
