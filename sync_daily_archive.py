import os
import io
import sys
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# --- КОНФИГУРАЦИЯ ---
SYMBOL = os.getenv("SYMBOL", "ADAUSDT").upper()
DAYS_BACK = int(os.getenv("DAYS_BACK", 3))  # Проверять последние N закрытых дней (не считая сегодня)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "movielogline/trade")

METRICS_NUMERIC_COLS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def download_and_convert_spot_day(symbol: str, date_str: str) -> str | None:
    """Скачивает суточный архив спотовых сделок Binance Vision и сохраняет во временный Parquet."""
    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    print(f"[Spot {date_str}] Запрос архива: {url}")

    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"[Spot {date_str}] Архив пока недоступен на Binance (HTTP {resp.status_code}).")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "trade_id",
                    "price",
                    "qty",
                    "quote_qty",
                    "time",
                    "is_buyer_maker",
                    "is_best_match",
                ],
            )

    if df.empty:
        print(f"[Spot {date_str}] Файл пустой, пропуск.")
        return None

    # Строго приводим типы 1-в-1 под схему потокового сборщика
    df["trade_id"] = df["trade_id"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["qty"] = df["qty"].astype("float64")
    df["quote_qty"] = df["quote_qty"].astype("float64")
    df["time"] = df["time"].astype("int64")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    df["is_best_match"] = df["is_best_match"].astype(bool)

    sample_time = df["time"].iloc[0]
    unit = "us" if sample_time > 10**15 else "ms"
    df["timestamp"] = pd.to_datetime(df["time"], unit=unit, utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    tmp_path = f"tmp_{symbol}_{date_str}_spot_full.parquet"
    df.to_parquet(tmp_path, compression="snappy", index=False)
    print(f"[Spot {date_str}] Сконвертировано тиков: {len(df):,}")
    return tmp_path


def download_and_convert_metrics_day(symbol: str, date_str: str) -> str | None:
    """Скачивает суточный архив фьючерсных 5m метрик (OI + L/S) с Binance Vision."""
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    print(f"[Metrics {date_str}] Запрос архива: {url}")

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"[Metrics {date_str}] Архив пока недоступен на Binance (HTTP {resp.status_code}).")
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    if df.empty:
        print(f"[Metrics {date_str}] Файл пустой, пропуск.")
        return None

    df.columns = [c.strip() for c in df.columns]
    for col in METRICS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    tmp_path = f"tmp_{symbol}_{date_str}_metrics_full.parquet"
    df.to_parquet(tmp_path, compression="snappy", index=False)
    print(f"[Metrics {date_str}] Сконвертировано 5m срезов: {len(df):,}")
    return tmp_path


def sync_spot_closed_days(api: HfApi, repo_files: set[str], now_utc: datetime):
    """Синхронизирует спотовые тики и заменяет потоковые чанки эталонным full.parquet."""
    print(f"\n--- 1. Архивация спотовых тиков ({SYMBOL}) ---")
    for days_ago in range(1, DAYS_BACK + 1):
        date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        day_prefix = f"ticks/{SYMBOL}/{date_str}/"
        full_repo_path = f"{day_prefix}full.parquet"

        existing_in_day = [f for f in repo_files if f.startswith(day_prefix)]

        # Если эталонный full.parquet уже есть
        if full_repo_path in repo_files:
            stray_chunks = [f for f in existing_in_day if f != full_repo_path]
            if not stray_chunks:
                print(f"[Spot {date_str}] Уже заархивирован (full.parquet на месте). Пропуск.")
                continue

            print(f"[Spot {date_str}] Удаление {len(stray_chunks)} лишних чанков...")
            ops = [CommitOperationDelete(path_in_repo=f) for f in stray_chunks]
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Cleanup stray chunks for {SYMBOL} {date_str}",
            )
            continue

        tmp_parquet = None
        try:
            tmp_parquet = download_and_convert_spot_day(SYMBOL, date_str)
            if not tmp_parquet:
                continue

            operations = []
            # Удаляем все старые чанки этого дня и кладём единый full.parquet
            if existing_in_day:
                print(f"[Spot {date_str}] Замена {len(existing_in_day)} потоковых чанков на единый full.parquet...")
                for old_file in existing_in_day:
                    operations.append(CommitOperationDelete(path_in_repo=old_file))

            operations.append(
                CommitOperationAdd(path_in_repo=full_repo_path, path_or_fileobj=tmp_parquet)
            )

            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive full day {SYMBOL} {date_str} (replace chunks)",
            )
            print(f"[Spot {date_str} ✓] Успешно заархивировано в {HF_REPO_ID}/{full_repo_path}")
        except Exception as e:
            print(f"[Spot {date_str} !] Ошибка при обработке дня: {e}")
        finally:
            if tmp_parquet and os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)


def sync_futures_metrics_days(api: HfApi, repo_files: set[str], now_utc: datetime):
    """Докачивает закрытые суточные архивы фьючерсных метрик (Open Interest + L/S Ratio)."""
    print(f"\n--- 2. Архивация фьючерсных метрик OI + L/S ({SYMBOL}) ---")
    operations = []
    temp_files = []

    try:
        for days_ago in range(1, DAYS_BACK + 1):
            date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            repo_path = f"metrics/{SYMBOL}/{date_str}/full.parquet"

            if repo_path in repo_files:
                print(f"[Metrics {date_str}] Уже заархивирован. Пропуск.")
                continue

            try:
                tmp_parquet = download_and_convert_metrics_day(SYMBOL, date_str)
                if tmp_parquet:
                    temp_files.append(tmp_parquet)
                    operations.append(
                        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=tmp_parquet)
                    )
            except Exception as e:
                print(f"[Metrics {date_str} !] Ошибка скачивания/конвертации: {e}")

        if operations:
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive futures OI metrics for {SYMBOL} ({len(operations)} days)",
            )
            print(f"[Metrics ✓] Загружено дней в {HF_REPO_ID}: {len(operations)}")
        else:
            print("[Metrics =] Все доступные закрытые дни уже в архиве.")
    finally:
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)


def sync_closed_days():
    if not HF_TOKEN or not HF_REPO_ID:
        raise RuntimeError("Не заданы переменные окружения HF_TOKEN или HF_REPO_ID!")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    # Получаем актуальный список файлов в датасете на Hugging Face один раз
    repo_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    now_utc = datetime.now(timezone.utc)

    sync_spot_closed_days(api, repo_files, now_utc)
    sync_futures_metrics_days(api, repo_files, now_utc)


if __name__ == "__main__":
    print(f"=== Старт авто-архиватора для {SYMBOL} (окно: {DAYS_BACK} дн.) ===")
    sync_closed_days()
    print("=== Архивация завершена ===")
