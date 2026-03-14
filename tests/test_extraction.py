from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction import (
    extract_date,
    extract_fields,
    extract_total,
    extract_vendor,
    normalize_date,
    _normalize_amount,
)


class TestNormalizeDate:

    def test_iso_format(self):
        assert normalize_date("2024-01-15") == "2024-01-15"

    def test_slash_dmy(self):
        assert normalize_date("15/01/2024") == "2024-01-15"

    def test_slash_ymd(self):
        assert normalize_date("2024/01/15") == "2024-01-15"

    def test_dot_dmy(self):
        assert normalize_date("15.01.2024") == "2024-01-15"

    def test_ambiguous_prefers_dmy(self):
        result = normalize_date("01/06/2024")
        assert result == "2024-06-01"

    def test_unambiguous_day_gt_12(self):
        assert normalize_date("25/06/2024") == "2024-06-25"

    def test_unambiguous_month_gt_12(self):
        assert normalize_date("06/25/2024") == "2024-06-25"

    def test_two_digit_year(self):
        result = normalize_date("15/01/24")
        assert result == "2024-01-15"

    def test_named_month(self):
        assert normalize_date("15 Jan 2024") == "2024-01-15"

    def test_named_month_full(self):
        assert normalize_date("15 January 2024") == "2024-01-15"

    def test_compact_ymd(self):
        assert normalize_date("20240115") == "2024-01-15"

    def test_out_of_range_returns_none(self):
        assert normalize_date("notadate") is None

    def test_year_too_old(self):
        assert normalize_date("01/01/1990") is None


class TestNormalizeAmount:

    def test_simple(self):
        assert _normalize_amount("10.50") == 10.50

    def test_comma_decimal(self):
        assert _normalize_amount("10,50") == 10.50

    def test_thousands_comma(self):
        assert _normalize_amount("1,234.56") == 1234.56

    def test_european_thousands(self):
        assert _normalize_amount("1.234,56") == 1234.56

    def test_invalid(self):
        assert _normalize_amount("abc") is None


class TestExtractDate:

    def test_date_with_keyword(self):
        text = "STORE NAME\nDate: 15/01/2024\nExpiry: 20/01/2025"
        result = extract_date(text)
        assert result == "2024-01-15"

    def test_date_without_keyword(self):
        text = "STORE\n15/01/2024\nTOTAL 10.00"
        result = extract_date(text)
        assert result == "2024-01-15"

    def test_no_date(self):
        assert extract_date("No date here") is None


class TestExtractTotal:

    def test_total_keyword(self):
        text = "Item A 5.00\nItem B 3.00\nTOTAL 8.00"
        assert extract_total(text) == "8.00"

    def test_grand_total(self):
        text = "Subtotal 7.00\nTax 1.00\nGrand Total 8.00"
        assert extract_total(text) == "8.00"

    def test_total_payable(self):
        text = "Subtotal 7.00\nTotal Payable: RM 8.00"
        assert extract_total(text) == "8.00"

    def test_fallback_bottom_up(self):
        text = "Item 5.00\nItem 3.00\n8.00"
        assert extract_total(text) == "8.00"

    def test_excludes_cash(self):
        text = "TOTAL 8.00\nCASH 10.00\nCHANGE 2.00"
        assert extract_total(text) == "8.00"

    def test_currency_symbol(self):
        text = "TOTAL $25.50"
        assert extract_total(text) == "25.50"

    def test_thousand_separator(self):
        text = "TOTAL RM 1,234.56"
        assert extract_total(text) == "1234.56"

    def test_no_total(self):
        assert extract_total("No amounts here") is None


class TestExtractVendor:

    def test_first_line_vendor(self):
        text = "GARDENIA BAKERIES\nDate: 19/08/2017\nTotal: 50.60"
        assert extract_vendor(text) == "GARDENIA BAKERIES"

    def test_skips_phone_number(self):
        text = "+60 123 456 789\nMY STORE\nDate: 01/01/2024"
        assert extract_vendor(text) == "MY STORE"

    def test_skips_date_line(self):
        text = "01/01/2024\nSHOP NAME\nTotal: 10.00"
        assert extract_vendor(text) == "SHOP NAME"

    def test_prefers_uppercase(self):
        text = "some info line\nMY STORE SDN BHD\nmore info"
        vendor = extract_vendor(text)
        assert vendor == "MY STORE SDN BHD"

    def test_returns_none_for_empty(self):
        assert extract_vendor("") is None

    def test_skips_address_line(self):
        text = "123 Jalan Bukit Bintang\nSHOP NAME\nTotal: 10.00"
        assert extract_vendor(text) == "SHOP NAME"

    def test_skips_tel_line(self):
        text = "Tel: 03-12345678\nSHOP NAME\nTotal: 10.00"
        assert extract_vendor(text) == "SHOP NAME"


class TestExtractFields:

    def test_full_receipt(self):
        text = "ACME CORP\nDate: 2024-01-01\nItem A 5.00\nTOTAL 10.00"
        fields = extract_fields(text)
        assert fields["vendor"] == "ACME CORP"
        assert fields["date"] == "2024-01-01"
        assert fields["total"] == "10.00"

    def test_missing_fields(self):
        text = "No structured data"
        fields = extract_fields(text)
        assert fields["vendor"] is None or isinstance(fields["vendor"], str)
        assert fields["date"] is None
        assert fields["total"] is None
