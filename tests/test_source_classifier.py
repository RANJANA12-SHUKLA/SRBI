import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval import classify_source


def test_classify_source_urls():
    company_domain = "examplemanufacturing.com"
    cases = {
        "https://examplemanufacturing.com/about": "PRIMARY",
        "https://www.examplemanufacturing.com/investors": "PRIMARY",
        "https://www.bse.com/stock-share-price/example": "PRIMARY",
        "https://www.nseindia.com/get-quotes/equity?symbol=EXAMPLE": "PRIMARY",
        "https://www.linkedin.com/company/example-manufacturing/": "PRIMARY",
        "https://www.linkedin.com/in/jane-doe/": "SECONDARY",
        "https://techcrunch.com/2026/01/01/example-funding/": "SECONDARY",
        "https://yourstory.com/2026/02/example-expansion": "SECONDARY",
        "https://www.crunchbase.com/organization/example-manufacturing": "SECONDARY",
        "https://unknownindustryblog.com/example-plant-tour": "SECONDARY",
    }

    for url, expected in cases.items():
        assert classify_source(url, company_domain) == expected
