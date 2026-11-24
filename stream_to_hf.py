import os
import json
import time
import shutil
import asyncio
import websockets
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import HfApi

# --- НАСТРОЙКИ ---
SYMBOL = os.getenv("SYMBOL", "adausdt").lower()
WS_URL = f"wss://data-stream.binance.vision:443/ws/{SYMBOL}@trade"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "movielogline/trade")




# Таймеры (в секундах)
RUN_DURATION_SEC = int(os.getenv("RUN_SECONDS", 5 * 3600))   # 5 часов = 18000 сек
FLUSH_INTERVAL_SEC = int(os.getenv("FLUSH_SECONDS", 600))    # Сброс на HF каждые 10 минут
MAX_BUFFER_TICKS = 50_000                                    # Или принудительный сброс при 50k тиков

STAGE_DIR = "hf_stage"


def flush_to_huggingface(buffer: list, api: HfApi):
    """Сохраняет накопленный буфер в Parquet по датам UTC и отправляет на HF."""
    if not buffer:
        return

    df = pd.DataFrame(buffer)
    buffer.clear()

    # Формируем колонку даты UTC для раскладки по папкам
    df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    chunk_time = datetime.now(timezone.utc).strftime("%H%M%S")

    if os.path.exists(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)

    created_files = 0
    for date_str, group in df.groupby("date_str"):
        out_dir = os.path.join(STAGE_DIR, "ticks", SYMBOL.upper(), date_str)
        os.makedirs(out_dir, exist_ok=True)

        file_path = os.path.join(out_dir, f"chunk_{chunk_time}.parquet")
        group.drop(columns=["date_str"]).to_parquet(file_path, compression="snappy", index=False)
        created_files += 1

    print(f"[{chunk_time} UTC] Подготовлено тиков: {len(df)} (файлов: {created_files}). Загрузка на HF...")

    try:
        api.upload_folder(
            folder_path=STAGE_DIR,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Stream {SYMBOL.upper()} {chunk_time} UTC ({len(df)} ticks)",
        )
        print(f"[HuggingFace] Успешно загружено в {HF_REPO_ID}")
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
    except Exception as e:
        print(f"[!] Ошибка загрузки на HuggingFace: {e}")


async def run_streamer():
    if not HF_TOKEN or not HF_REPO_ID:
        raise RuntimeError("Не заданы переменные окружения HF_TOKEN или HF_REPO_ID!")

    api = HfApi(token=HF_TOKEN)
    buffer = []

    start_time = time.monotonic()
    deadline = start_time + RUN_DURATION_SEC
    last_flush = start_time

    print(f"[Start] Символ: {SYMBOL.upper()} | Эндпоинт: data-stream.binance.vision")
    print(f"[Timer] Авто-закрытие через {RUN_DURATION_SEC / 3600:.2f} ч. | Сброс каждые {FLUSH_INTERVAL_SEC} сек.")

    try:
        while time.monotonic() < deadline:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] WebSocket подключен.")

                    while time.monotonic() < deadline:
                        # Ждем сообщение не дольше 5 сек, чтобы регулярно проверять таймеры
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(raw_msg)

                            t_raw = data.get("T")
                            unit = "us" if t_raw > 10**15 else "ms"
                            price = float(data["p"])
                            qty = float(data["q"])

                            buffer.append({
                                "trade_id": int(data["t"]),
                                "price": price,
                                "qty": qty,
                                "quote_qty": price * qty,
                                "time": int(t_raw),
                                "is_buyer_maker": bool(data["m"]),
                                "is_best_match": bool(data.get("M", True)),
                                "timestamp": pd.to_datetime(t_raw, unit=unit, utc=True),
                            })
                        except asyncio.TimeoutError:
                            pass

                        now = time.monotonic()
                        # Сброс по времени или по переполнению буфера
                        if buffer and (
                            (now - last_flush >= FLUSH_INTERVAL_SEC)
                            or (len(buffer) >= MAX_BUFFER_TICKS)
                        ):
                            batch = buffer.copy()
                            buffer.clear()
                            last_flush = now
                            await asyncio.to_thread(flush_to_huggingface, batch, api)

            except Exception as e:
                if time.monotonic() >= deadline:
                    break
                print(f"[!] Сбой WebSocket: {e}. Переподключение через 5 сек...")
                await asyncio.sleep(5)
    finally:
        # Финальный сброс остатков буфера перед закрытием по 5-часовому таймеру
        if buffer:
            print("[Shutdown] Сброс финального хвоста тиков перед завершением...")
            await asyncio.to_thread(flush_to_huggingface, buffer, api)
        print("[✓] 5-часовой таймер завершен штатно. Выход.")


if __name__ == "__main__":
    asyncio.run(run_streamer())
