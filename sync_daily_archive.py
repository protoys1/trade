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


def download_and_convert_day(symbol: str, date_str: str) -> str | None:
    """Скачивает суточный архив сделок Binance Vision и сохраняет во временный Parquet."""
    url = f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    print(f"[{date_str}] Запрос архива: {url}")

    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"[{date_str}] Архив пока недоступен на Binance (HTTP {resp.status_code}).")
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
        print(f"[{date_str}] Файл пустой, пропуск.")
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

    tmp_path = f"tmp_{symbol}_{date_str}_full.parquet"
    df.to_parquet(tmp_path, compression="snappy", index=False)
    print(f"[{date_str}] Сконвертировано тиков: {len(df):,}")
    return tmp_path


def sync_closed_days():
    if not HF_TOKEN or not HF_REPO_ID:
        raise RuntimeError("Не заданы переменные окружения HF_TOKEN или HF_REPO_ID!")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)

    # Получаем актуальный список файлов в датасете на Hugging Face
    repo_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    now_utc = datetime.now(timezone.utc)

    # Проверяем только закрытые дни (от yesterday до DAYS_BACK дней назад). Сегодня не трогаем!
    for days_ago in range(1, DAYS_BACK + 1):
        date_str = (now_utc - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        day_prefix = f"ticks/{SYMBOL}/{date_str}/"
        full_repo_path = f"{day_prefix}full.parquet"

        existing_in_day = [f for f in repo_files if f.startswith(day_prefix)]

        # Если эталонный full.parquet уже есть
        if full_repo_path in repo_files:
            stray_chunks = [f for f in existing_in_day if f != full_repo_path]
            if not stray_chunks:
                print(f"[{date_str}] Уже заархивирован (full.parquet на месте). Пропуск.")
                continue

            # Если вдруг остались лишние чанки рядом с full.parquet — вычищаем их
            print(f"[{date_str}] Удаление {len(stray_chunks)} лишних чанков...")
            ops = [CommitOperationDelete(path_in_repo=f) for f in stray_chunks]
            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Cleanup stray chunks for {SYMBOL} {date_str}",
            )
            continue

        # Скачиваем официальный архив дня с Binance Vision
        tmp_parquet = download_and_convert_day(SYMBOL, date_str)
        if not tmp_parquet:
            continue

        try:
            operations = []
            # Если за этот день были чанки стрима — сносим всю папку дня и сразу кладём full.parquet
            if existing_in_day:
                print(f"[{date_str}] Замена {len(existing_in_day)} потоковых чанков на единый full.parquet...")
                operations.append(CommitOperationDelete(path_in_repo=day_prefix, is_folder=True))

            operations.append(
                CommitOperationAdd(path_in_repo=full_repo_path, path_or_fileobj=tmp_parquet)
            )

            api.create_commit(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Archive full day {SYMBOL} {date_str} (replace chunks)",
            )
            print(f"[{date_str} ✓] Успешно заархивировано в {HF_REPO_ID}/{full_repo_path}")
        finally:
            if os.path.exists(tmp_parquet):
                os.remove(tmp_parquet)


if __name__ == "__main__":
    print(f"=== Старт авто-архиватора для {SYMBOL} (окно: {DAYS_BACK} дн.) ===")
    sync_closed_days()
    print("=== Архивация завершена ===")