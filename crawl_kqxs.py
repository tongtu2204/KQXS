"""Crawl kết quả giải đặc biệt miền Bắc từ thongkemienbac.com."""

from datetime import datetime
from pathlib import Path
from time import sleep

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://thongkemienbac.com/thong-ke-giai-dac-biet-nam-{}.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def crawl_year(year: int) -> list[dict]:
    url = BASE_URL.format(year)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.select_one("table#dacbietbyyear")
    if table is None:
        raise RuntimeError(f"Không tìm thấy bảng dữ liệu năm {year}")

    rows = []
    for cell in table.select("td[title]"):
        prefix = cell.select_one("span.db-prefix")
        last_two = cell.select_one("span.colorRed")
        if last_two is None:
            continue

        prefix_text = prefix.get_text(strip=True) if prefix else ""
        last_two_text = last_two.get_text(strip=True).zfill(2)
        rows.append(
            {
                "date": cell["title"],
                "full_result": prefix_text + last_two_text,
                "last_2_digits": last_two_text,
            }
        )
    return rows


def main(start_year: int = 2002) -> None:
    rows = []
    end_year = datetime.now().year

    for year in range(start_year, end_year + 1):
        year_rows = crawl_year(year)
        rows.extend(year_rows)
        print(f"{year}: {len(year_rows)} kết quả")
        sleep(1)

    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"], format="%d-%m-%Y")
    result = (
        result.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    result.insert(1, "year", result["date"].dt.year)
    result.insert(2, "month", result["date"].dt.month)
    result.insert(3, "day", result["date"].dt.day)

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"kqxsmb_2002_{end_year}.csv"
    result.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nTổng số kỳ: {len(result)}")
    print(f"Đã lưu: {output_file}")


if __name__ == "__main__":
    main()
