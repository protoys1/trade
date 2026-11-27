import os
import io
import sys
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, hf_hub_download

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# --- КОНФИГУРАЦИЯ ---
SYMBOL = os.getenv("SYMBOL", "ADAUSDT").upper()
DAYS_BACK = int(os.getenv("DAYS_BACK", 3))  # Глубина проверки закрытых дней

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


# =============================================================================
# 1. СПОТОВЫЕ ТИКИ (СДЕЛКИ)
# =============================================================================
def download_and_convert_spot_day(symbol: str, date_str: str) -> str | None:
    """Скачивает официальный суточный ZIP спотовых сделок с Binance Vision."""
    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    print(f"[Spot {date_str}] Запрос официального архива: {url}")

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            print(f"[Spot {date_str}] Официальный архив пока недоступен (HTTP {resp.status_code}).")
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
            return None

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

        tmp_path = f"tmp_{symbol}_{date_str}_spot_vision.parquet"
        df.to_parquet(tmp_path, compression="snappy", index=False)
        print(f"[Spot {date_str}] Официальный архив сконвертирован: {len(df):,} сделок")
        return tmp_path
    except Exception as e:
        print(f"[Spot {date_str} !] Ошибка при скачивании: {e}")
        return None


def merge_existing_chunks_fallback(symbol: str, date_str: str, chunk_files: list[str]) -> str | None:
    """Fallback: склеивает потоковые чанки из репозитория в один full.parquet."""
    print(f"[Spot {date_str}] Fallback-склейка {len(chunk_files)} мелких чанков из HF...")
    dfs = []
    for repo_file in sorted(chunk_files):
        try:
            local_p = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=repo_file, token=HF_TOKEN)
            part_df = pd.read_parquet(local_p)
            if not part_df.empty:
                dfs.append(part_df)
        except Exception as e:
            print(f"  [!] Ошибка чтения чанка {repo_file}: {e}")

    if not dfs:
        return None

    full_df = pd.concat(dfs, ignore_index=True)
    if "trade_id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["trade_id"]).sort_values("time")
    else:
        full_df = full_df.sort_values("time")

    tmp_path = f"tmp_{symbol}_{date_str}_spot_merged.parquet"
    full_df.to_parquet(tmp_path, compression="snappy", index=False)
    print(f"[Spot {date_str} ✓] Склеено: {len(full_df):,} уникальных тиков")
    return tmp_path


def sync_spot_closed_days(api: HfApi, repo_files: set[str], now_utc: datetime):
    """Синхронизирует спотовые тики: удаляет чанки и оставляет строго full.parquet."""
    print(f"\n--- 1. Архивация спотовых тиков ({SYMBOL}) ---")
    for days_ago in range(1, DAYS_BACK + 1):
        date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        day_prefix = f"ticks/{SYMBOL}/{date_str}/"
        full_repo_path = f"{day_prefix}full.parquet"

        existing_in_day = [f for f in repo_files if f.startswith(day_prefix)]
        stray_chunks = [f for f in existing_in_day if f != full_repo_path]

        # Если full.parquet уже есть
        if full_repo_path in repo_files:
            if not stray_chunks:
                print(f"[Spot {date_str}] Чисто (только full.parquet). Пропуск.")
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

        # full.parquet ещё нет — пробуем Binance Vision, при неудаче склеиваем накопленные чанки
        tmp_parquet = download_and_convert_spot_day(SYMBOL, date_str)
        if not tmp_parquet and stray_chunks:
            tmp_parquet = merge_existing_chunks_fallback(SYMBOL, date_str, stray_chunks)

        if not tmp_parquet:
            print(f"[Spot {date_str} -] Нет данных ни на Binance, ни в репозитории. Пропуск.")
            continue

        try:
            operations = []
            if stray_chunks:
                print(f"[Spot {date_str}] Замена {len(stray_chunks)} чанков на full.parquet...")
                for old_f in stray_chunks:
                    operations.append(CommitOperationDelete(path_in_repo=old_f))

            operations.append(CommitOperationAdd(path_in_repo=full_repo_path, path_or_fileobj=tmp_parquet))
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive full day {SYMBOL} {date_str} (replace chunks with full.parquet)",
            )
            print(f"[Spot {date_str} ✓] Заархивировано в {full_repo_path}")
        except Exception as e:
            print(f"[Spot {date_str} !] Ошибка коммита: {e}")
        finally:
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)


# =============================================================================
# 2. ФЬЮЧЕРСНЫЕ ТИКИ (СДЕЛКИ И ЛИКВИДАЦИИ USDT-M)
# =============================================================================
def download_and_convert_futures_trades_day(symbol: str, date_str: str) -> str | None:
    """Скачивает суточный архив фьючерсных сделок (USDT-M Futures trades) с Binance Vision."""
    url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    print(f"[Futures Trades {date_str}] Запрос архива сделок: {url}")

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            print(f"[Futures Trades {date_str}] Архив пока недоступен на Binance (HTTP {resp.status_code}).")
            return None

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Структура CSV на фьючерсах Binance: id, price, qty, quote_qty, time, is_buyer_maker
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
                    ],
                )

        if df.empty:
            return None

        df["trade_id"] = df["trade_id"].astype("int64")
        df["price"] = df["price"].astype("float64")
        df["qty"] = df["qty"].astype("float64")
        df["quote_qty"] = df["quote_qty"].astype("float64")
        df["time"] = df["time"].astype("int64")
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        sample_time = df["time"].iloc[0]
        unit = "us" if sample_time > 10**15 else "ms"
        df["timestamp"] = pd.to_datetime(df["time"], unit=unit, utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

        tmp_path = f"tmp_{symbol}_{date_str}_futures_trades.parquet"
        df.to_parquet(tmp_path, compression="snappy", index=False)
        print(f"[Futures Trades {date_str}] Сконвертировано фьючерсных сделок: {len(df):,}")
        return tmp_path
    except Exception as e:
        print(f"[Futures Trades {date_str} !] Ошибка: {e}")
        return None


def sync_futures_trades_days(api: HfApi, repo_files: set[str], now_utc: datetime):
    """Синхронизирует тики фьючерсных сделок (USDT-M)."""
    print(f"\n--- 2. Архивация фьючерсных сделок/тиков ({SYMBOL}) ---")
    for days_ago in range(1, DAYS_BACK + 1):
        date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        repo_path = f"futures_ticks/{SYMBOL}/{date_str}/full.parquet"

        if repo_path in repo_files:
            print(f"[Futures Trades {date_str}] Уже заархивирован. Пропуск.")
            continue

        tmp_parquet = download_and_convert_futures_trades_day(SYMBOL, date_str)
        if not tmp_parquet:
            continue

        try:
            operations = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=tmp_parquet)]
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive full day futures trades {SYMBOL} {date_str}",
            )
            print(f"[Futures Trades {date_str} ✓] Загружено в {repo_path}")
        except Exception as e:
            print(f"[Futures Trades {date_str} !] Ошибка коммита: {e}")
        finally:
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)


# =============================================================================
# 3. ФЬЮЧЕРСНЫЕ МЕТРИКИ (OI + L/S RATIO)
# =============================================================================
def download_and_convert_metrics_day(symbol: str, date_str: str) -> str | None:
    """Скачивает суточный архив 5m метрик фьючерсов (OI, L/S) с Binance Vision."""
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{date_str}.zip"
    print(f"[Metrics {date_str}] Запрос архива метрик: {url}")

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"[Metrics {date_str}] Архив пока недоступен на Binance (HTTP {resp.status_code}).")
            return None

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        if df.empty:
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
    except Exception as e:
        print(f"[Metrics {date_str} !] Ошибка: {e}")
        return None


def sync_futures_metrics_days(api: HfApi, repo_files: set[str], now_utc: datetime):
    """Синхронизирует фьючерсные метрики (OI + Long/Short Ratio)."""
    print(f"\n--- 3. Архивация фьючерсных метрик OI + L/S ({SYMBOL}) ---")
    operations = []
    temp_files = []

    try:
        for days_ago in range(1, DAYS_BACK + 1):
            date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            repo_path = f"metrics/{SYMBOL}/{date_str}/full.parquet"

            if repo_path in repo_files:
                print(f"[Metrics {date_str}] Уже заархивирован. Пропуск.")
                continue

            tmp_parquet = download_and_convert_metrics_day(SYMBOL, date_str)
            if tmp_parquet:
                temp_files.append(tmp_parquet)
                operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=tmp_parquet))

        if operations:
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive futures OI metrics for {SYMBOL} ({len(operations)} days)",
            )
            print(f"[Metrics ✓] Загружено дней в {HF_REPO_ID}: {len(operations)}")
        else:
            print("[Metrics =] Все доступные метрики уже в архиве.")
    finally:
        for p in temp_files:
            if os.path.exists(p):
                os.remove(p)


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
def sync_closed_days():
    if not HF_TOKEN or not HF_REPO_ID:
        raise RuntimeError("Не заданы переменные окружения HF_TOKEN или HF_REPO_ID!")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    repo_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    now_utc = datetime.now(timezone.utc)

    # 1. Спотовые сделки (чистим чанки, кладём full.parquet)
    sync_spot_closed_days(api, repo_files, now_utc)

    # 2. Фьючерсные сделки USDT-M (полные тики ликвидаций и плечей)
    sync_futures_trades_days(api, repo_files, now_utc)

    # 3. Фьючерсные метрики (Open Interest + Long/Short)
    sync_futures_metrics_days(api, repo_files, now_utc)


if __name__ == "__main__":
    print(f"=== Старт авто-архиватора для {SYMBOL} (глубина: {DAYS_BACK} дн.) ===")
    sync_closed_days()
    print("=== Архивация успешно завершена ===")
