import os
import gc
import sys
import json
import time
import signal
import shutil
import asyncio
import websockets
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import HfApi, CommitOperationAdd

# Отключаем лишние кэш-прогрессы HF на раннере и включаем мгновенный вывод логов
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# --- НАСТРОЙКИ ---
SYMBOL = os.getenv("SYMBOL", "adausdt").lower()
WS_URL = f"wss://data-stream.binance.vision:443/ws/{SYMBOL}@trade"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "movielogline/trade")

# Короткие 55-минутные смены — не ловят обрыв эфемерных VM GitHub Actions
RUN_DURATION_SEC = int(os.getenv("RUN_SECONDS", 3300))     # 55 минут = 3300 сек
FLUSH_INTERVAL_SEC = int(os.getenv("FLUSH_SECONDS", 600))  # Сброс каждые 10 минут
MAX_BUFFER_TICKS = 50_000

STAGE_DIR = "hf_stage"


def flush_to_huggingface(batch: list, api: HfApi) -> bool:
    """Конвертирует накопленный буфер в Parquet по датам UTC и отправляет атомарным коммитом на HF."""
    if not batch:
        return True

    df = pd.DataFrame(batch)
    unit = "us" if int(df["time"].iloc[0]) > 10**15 else "ms"
    df["timestamp"] = pd.to_datetime(df["time"], unit=unit, utc=True)
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    chunk_time = datetime.now(timezone.utc).strftime("%H%M%S")

    if os.path.exists(STAGE_DIR):
        shutil.rmtree(STAGE_DIR, ignore_errors=True)

    operations = []
    for date_str, group in df.groupby("date_str"):
        out_dir = os.path.join(STAGE_DIR, "ticks", SYMBOL.upper(), date_str)
        os.makedirs(out_dir, exist_ok=True)
        local_path = os.path.join(out_dir, f"chunk_{chunk_time}.parquet")
        repo_path = f"ticks/{SYMBOL.upper()}/{date_str}/chunk_{chunk_time}.parquet"

        group.drop(columns=["date_str"]).to_parquet(local_path, compression="snappy", index=False)
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path))

    tick_count = len(df)
    file_count = len(operations)
    del df
    gc.collect()

    print(f"[{chunk_time} UTC] Собрано тиков: {tick_count} (файлов: {file_count}). Отправка в {HF_REPO_ID}...")

    try:
        api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Stream {SYMBOL.upper()} {chunk_time} UTC ({tick_count} ticks)",
        )
        print(f"[HuggingFace ✓] Успешно загружено в {HF_REPO_ID}!")
        return True
    except Exception as e:
        print(f"[!] Ошибка загрузки на HuggingFace: {e}")
        return False
    finally:
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        gc.collect()


async def run_streamer():
    if not HF_TOKEN or not HF_REPO_ID:
        raise RuntimeError("Не задана переменная окружения HF_TOKEN!")

    api = HfApi(token=HF_TOKEN)
    user = api.whoami()
    print(f"[HF ✓] Токен принят! Аккаунт: {user['name']}")
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"[HF ✓] Датасет '{HF_REPO_ID}' готов к приёму файлов.")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # На Windows пропускаем

    buffer = []
    start_time = time.monotonic()
    deadline = start_time + RUN_DURATION_SEC
    last_flush = start_time
    flush_task: asyncio.Task | None = None

    async def flush_in_background(batch_to_send: list):
        ok = await asyncio.to_thread(flush_to_huggingface, batch_to_send, api)
        if not ok and len(buffer) < MAX_BUFFER_TICKS:
            # Возвращаем несброшенные тики в начало буфера, не раздувая память выше лимита
            space_left = MAX_BUFFER_TICKS - len(buffer)
            buffer[:0] = batch_to_send[-space_left:]

    print(
        f"[Start] Символ: {SYMBOL.upper()} | Смена: {RUN_DURATION_SEC / 60:.0f} мин. "
        f"| Сброс каждые {FLUSH_INTERVAL_SEC} сек."
    )

    try:
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=20, ping_timeout=20, close_timeout=5
                ) as ws:
                    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] WebSocket подключен, поток идёт...")

                    while not stop_event.is_set() and time.monotonic() < deadline:
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            data = json.loads(raw_msg)
                            price = float(data["p"])
                            qty = float(data["q"])

                            buffer.append({
                                "trade_id": int(data["t"]),
                                "price": price,
                                "qty": qty,
                                "quote_qty": price * qty,
                                "time": int(data["T"]),
                                "is_buyer_maker": bool(data["m"]),
                                "is_best_match": bool(data.get("M", True)),
                            })
                        except asyncio.TimeoutError:
                            pass

                        now = time.monotonic()
                        if buffer and (
                            (now - last_flush >= FLUSH_INTERVAL_SEC)
                            or (len(buffer) >= MAX_BUFFER_TICKS)
                        ):
                            if flush_task is None or flush_task.done():
                                if flush_task is not None and flush_task.exception():
                                    print(f"[!] Фоновая ошибка сброса: {flush_task.exception()}")
                                batch = buffer
                                buffer = []
                                last_flush = now
                                flush_task = asyncio.create_task(flush_in_background(batch))

            except Exception as e:
                if stop_event.is_set() or time.monotonic() >= deadline:
                    break
                print(f"[!] Сбой WebSocket: {e}. Переподключение через 5 сек...")
                await asyncio.sleep(5)
    finally:
        if flush_task is not None and not flush_task.done():
            print("[Shutdown] Ожидание завершения фоновой отправки...")
            await flush_task
        if buffer:
            print("[Shutdown] Сброс финального хвоста тиков перед завершением...")
            await asyncio.to_thread(flush_to_huggingface, buffer, api)
        print("[✓] Смена завершена штатно.")


if __name__ == "__main__":
    asyncio.run(run_streamer())
