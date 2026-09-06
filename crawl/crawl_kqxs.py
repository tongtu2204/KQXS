"""Crawl toàn bộ kết quả xổ số truyền thống miền Bắc từ năm 2020."""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from lxml import html


BASE_URL = "https://mketqua.net/so-ket-qua"
SOURCE_URL = BASE_URL
PROVINCE = "mb"

REQUEST_DAYS = 30
SLEEP_SECONDS = 1.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
}

PRIZE_ORDER = {
    "Đặc biệt": 1,
    "Giải nhất": 2,
    "Giải nhì": 3,
    "Giải ba": 4,
    "Giải tư": 5,
    "Giải năm": 6,
    "Giải sáu": 7,
    "Giải bảy": 8,
}

EXPECTED_NUMBERS = {
    "Đặc biệt": 1,
    "Giải nhất": 1,
    "Giải nhì": 2,
    "Giải ba": 6,
    "Giải tư": 4,
    "Giải năm": 6,
    "Giải sáu": 3,
    "Giải bảy": 4,
}

NUMBER_LENGTH = {
    "Đặc biệt": 5,
    "Giải nhất": 5,
    "Giải nhì": 5,
    "Giải ba": 5,
    "Giải tư": 4,
    "Giải năm": 4,
    "Giải sáu": 3,
    "Giải bảy": 2,
}

DATE_PATTERN = re.compile(r"(\d{1,2}-\d{1,2}-\d{4})")
NUMBER_PATTERN = re.compile(r"^\d+$")


def clean_text(value: str) -> str:
    """Chuẩn hóa khoảng trắng."""
    return " ".join(value.split())


def format_site_date(value: date) -> str:
    """Định dạng ngày theo format của website."""
    return f"{value.day}-{value.month}-{value.year}"


def parse_date(value: str) -> date:
    """Parse ngày từ format YYYY-MM-DD."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def extract_date(table) -> date:
    """Lấy ngày xổ số từ một bảng kết quả."""

    table_text = clean_text(table.text_content())
    match = DATE_PATTERN.search(table_text)

    if match is None:
        raise ValueError(
            "Không tìm thấy ngày trong bảng kết quả. "
            f"Nội dung bảng: {table_text[:300]}"
        )

    return datetime.strptime(
        match.group(1),
        "%d-%m-%Y",
    ).date()


def extract_prize_rows(
    table,
    result_date: date,
) -> list[dict]:
    """Lấy toàn bộ số của các giải trong một ngày."""

    rows = []

    html_rows = table.xpath("./tbody/tr")

    for row in html_rows:
        cells = row.xpath("./td")

        if len(cells) < 2:
            continue

        prize = clean_text(cells[0].text_content())

        if prize not in PRIZE_ORDER:
            continue

        number_nodes = cells[1].xpath(
            ".//div[@data-pattern]"
        )

        numbers = []

        for node in number_nodes:
            raw_number = node.get("data-sofar")

            if not raw_number:
                raw_number = clean_text(node.text_content())

            raw_number = str(raw_number).strip()

            if not NUMBER_PATTERN.fullmatch(raw_number):
                continue

            expected_length = NUMBER_LENGTH[prize]
            number = raw_number.zfill(expected_length)

            if len(number) != expected_length:
                raise ValueError(
                    f"Số không hợp lệ: "
                    f"date={result_date}, "
                    f"prize={prize}, "
                    f"number={raw_number}"
                )

            numbers.append(number)

        expected_count = EXPECTED_NUMBERS[prize]

        if len(numbers) != expected_count:
            raise ValueError(
                f"Số lượng kết quả không đúng: "
                f"date={result_date}, "
                f"prize={prize}, "
                f"expected={expected_count}, "
                f"actual={len(numbers)}"
            )

        for prize_index, number in enumerate(numbers, start=1):
            rows.append(
                {
                    "date": result_date,
                    "province": PROVINCE,
                    "prize": prize,
                    "prize_index": prize_index,
                    "number": number,
                    "last_2_digits": number[-2:],
                    "source": SOURCE_URL,
                    "crawled_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            )

    found_prizes = {row["prize"] for row in rows}
    expected_prizes = set(PRIZE_ORDER)

    if found_prizes != expected_prizes:
        missing_prizes = expected_prizes - found_prizes

        raise ValueError(
            f"Thiếu giải trong ngày {result_date}: "
            f"{sorted(missing_prizes)}"
        )

    return rows


def find_result_tables(document):
    """
    Tìm bảng kết quả theo cấu trúc XPath:

    div#result_mb
      div.tb-phoi-6
        div.color333
          table#result_tab_mb
    """

    xpath = (
        "//div[@id='result_mb']"
        "//div[contains("
        "concat(' ', normalize-space(@class), ' '), "
        "' tb-phoi-6 '"
        ")]"
        "//div[contains("
        "concat(' ', normalize-space(@class), ' '), "
        "' color333 '"
        ")]"
        "//table[@id='result_tab_mb']"
    )

    tables = document.xpath(xpath)

    if tables:
        return tables

    # Fallback nếu class trung gian thay đổi.
    return document.xpath(
        "//table[@id='result_tab_mb']"
    )


def validate_block_rows(
    rows: list[dict],
    start_date: date,
    end_date: date,
) -> None:
    """
    Kiểm tra dữ liệu trong block.

    Không bắt buộc mọi ngày lịch phải có kết quả,
    vì có các ngày xổ số nghỉ quay như dịp Tết.
    """

    if not rows:
        raise RuntimeError(
            f"Không có dữ liệu trong block "
            f"{start_date} -> {end_date}"
        )

    rows_by_date = {}

    for row in rows:
        rows_by_date.setdefault(row["date"], 0)
        rows_by_date[row["date"]] += 1

    invalid_dates = {
        result_date: row_count
        for result_date, row_count in rows_by_date.items()
        if row_count != 27
    }

    if invalid_dates:
        raise RuntimeError(
            "Có ngày không đủ 27 kết quả: "
            f"{invalid_dates}"
        )

    calendar_days = (
        end_date - start_date
    ).days + 1

    actual_days = len(rows_by_date)
    no_draw_days = calendar_days - actual_days

    if no_draw_days > 0:
        print(
            f"  Có {no_draw_days} ngày không quay "
            f"trong block này; tiếp tục bỏ qua."
        )


def crawl_period(
    session: requests.Session,
    start_date: date,
    end_date: date,
    request_timeout: int = 60,
    max_retries: int = 3,
) -> list[dict]:
    """Crawl một block tối đa 30 ngày."""

    requested_days = (
        end_date - start_date
    ).days + 1

    if requested_days < 1:
        raise ValueError("Khoảng ngày không hợp lệ.")

    request_count = min(
        requested_days,
        REQUEST_DAYS,
    )

    payload = {
        "code": PROVINCE,
        "date": format_site_date(end_date),
        "count": str(request_count),
        "dow": "7",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(
                BASE_URL,
                data=payload,
                headers=HEADERS,
                timeout=request_timeout,
            )

            response.raise_for_status()

            document = html.fromstring(
                response.content
            )

            tables = find_result_tables(document)

            if not tables:
                raise RuntimeError(
                    "Không tìm thấy bảng "
                    "result_tab_mb."
                )

            rows = []

            for table in tables:
                result_date = extract_date(table)

                # Chỉ lấy ngày nằm trong block hiện tại.
                if not start_date <= result_date <= end_date:
                    continue

                rows.extend(
                    extract_prize_rows(
                        table=table,
                        result_date=result_date,
                    )
                )

            validate_block_rows(
                rows=rows,
                start_date=start_date,
                end_date=end_date,
            )

            return rows

        except Exception as exc:
            last_error = exc

            if attempt < max_retries:
                wait_seconds = attempt * 3

                print(
                    f"  Lỗi lần {attempt}/{max_retries}: "
                    f"{exc}. "
                    f"Thử lại sau {wait_seconds} giây..."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Không thể crawl block "
        f"{start_date} -> {end_date}: "
        f"{last_error}"
    )


def crawl_all(
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Crawl ngược từng block 30 ngày về năm 2020."""

    if start_date > end_date:
        raise ValueError(
            "start_date phải nhỏ hơn hoặc bằng end_date."
        )

    all_rows = []

    total_calendar_days = (
        end_date - start_date
    ).days + 1

    completed_calendar_days = 0
    current_end = end_date

    with requests.Session() as session:
        while current_end >= start_date:
            current_start = max(
                start_date,
                current_end - timedelta(
                    days=REQUEST_DAYS - 1
                ),
            )

            block_calendar_days = (
                current_end - current_start
            ).days + 1

            print(
                f"Crawl {current_start} -> {current_end} "
                f"({block_calendar_days} ngày)..."
            )

            block_rows = crawl_period(
                session=session,
                start_date=current_start,
                end_date=current_end,
            )

            all_rows.extend(block_rows)

            completed_calendar_days += (
                block_calendar_days
            )

            block_actual_days = len(
                {
                    row["date"]
                    for row in block_rows
                }
            )

            print(
                f"  OK: {block_actual_days} ngày có kết quả | "
                f"{len(block_rows):,} dòng | "
                f"tiến độ lịch "
                f"{completed_calendar_days:,}/"
                f"{total_calendar_days:,} ngày"
            )

            # Lùi về block trước.
            current_end = current_start - timedelta(
                days=1
            )

            if current_end >= start_date:
                time.sleep(SLEEP_SECONDS)

    result = pd.DataFrame(all_rows)

    if result.empty:
        raise RuntimeError(
            "Không crawl được dữ liệu nào."
        )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = result.drop_duplicates(
        subset=[
            "date",
            "province",
            "prize",
            "prize_index",
        ]
    )

    result = result.sort_values(
        by=[
            "date",
            "prize",
            "prize_index",
        ],
        key=lambda column: (
            column.map(PRIZE_ORDER)
            if column.name == "prize"
            else column
        ),
    ).reset_index(drop=True)

    rows_by_date = (
        result.groupby(result["date"].dt.date)
        .size()
    )

    invalid_dates = rows_by_date[
        rows_by_date != 27
    ]

    if not invalid_dates.empty:
        raise RuntimeError(
            "Các ngày không có đúng 27 kết quả:\n"
            f"{invalid_dates}"
        )

    actual_days = len(rows_by_date)
    expected_rows = actual_days * 27

    if len(result) != expected_rows:
        raise RuntimeError(
            f"Số dòng không đúng: "
            f"expected={expected_rows:,}, "
            f"actual={len(result):,}"
        )

    no_draw_days = (
        total_calendar_days - actual_days
    )

    print(
        f"\nTổng ngày lịch: "
        f"{total_calendar_days:,}"
    )

    print(
        f"Ngày có kết quả: "
        f"{actual_days:,}"
    )

    print(
        f"Ngày không quay: "
        f"{no_draw_days:,}"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl toàn bộ kết quả xổ số "
            "miền Bắc từ năm 2007."
        )
    )

    parser.add_argument(
        "--start-date",
        default="2007-01-01",
        help="Ngày bắt đầu, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Ngày kết thúc, format YYYY-MM-DD.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Đường dẫn file CSV đầu ra.",
    )

    args = parser.parse_args()

    start_date = parse_date(
        args.start_date
    )

    end_date = parse_date(
        args.end_date
    )

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(
            f"data/raw/kqxsmb_all_prizes_"
            f"{start_date.year}_{end_date.year}.csv"
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = crawl_all(
        start_date=start_date,
        end_date=end_date,
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nHoàn tất crawl dữ liệu.")
    print(
        f"Số ngày có kết quả: "
        f"{result['date'].nunique():,}"
    )
    print(
        f"Số dòng: "
        f"{len(result):,}"
    )
    print(
        f"File đầu ra: {output_file}"
    )


if __name__ == "__main__":
    main()