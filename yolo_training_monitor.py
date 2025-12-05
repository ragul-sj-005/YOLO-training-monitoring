import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import signal
import psutil
import requests

# === Telegram Bot Config ===
bot_token = '7281041778:AAEZeY5sewcRsfhzAxzuGKi3CD3L55m8tuc'  # Replace with your bot token
chat_id = '1082059922'  # Replace with your chat ID
worker_name = "Muhammed Roshan S"

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {'chat_id': chat_id, 'text': message}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print("[ALERT] Telegram alert sent!")
        else:
            print(f"[ERROR] Telegram alert failed: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram alert: {e}")

# === Setup ===
BASE_DIR = 'C:/Users/HP/ultralytics/runs/detect'
EXP_NAME = sorted(glob.glob(os.path.join(BASE_DIR, "train*")), key=os.path.getmtime)[-1]
SAVE_DIR = os.path.join(BASE_DIR, os.path.basename(EXP_NAME))
RESULTS_CSV = os.path.join(SAVE_DIR, 'results.csv')
REPORT_PATH = os.path.join(SAVE_DIR, 'training_report.txt')

# === Shared state variables ===
insights_log = []
overfitting_counter = [0]
stopped = [False]
sent_overfit_alert = [False]
sent_lowmap_alert = [False]
sent_stagnant_alert = [False]
last_checked_epoch = [-1]  # ✅ NEW: Track last processed epoch

def find_yolo_process():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'yolo' in ' '.join(proc.info['cmdline']).lower():
                return proc.pid
        except Exception:
            continue
    return None

print(f"[INFO] Watching: {RESULTS_CSV}")

# === Plot Setup ===
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("YOLOv8 Live Training Monitor", fontsize=16)
fig.subplots_adjust(bottom=0.4)  # Increased bottom spacing

train_line, = ax1.plot([], [], 'r-', label='Train Loss')
val_line, = ax1.plot([], [], 'b-', label='Val Loss')
map_line, = ax2.plot([], [], 'g-', label='mAP@0.5')

ax1.set_ylabel("Loss")
ax2.set_ylabel("mAP")
ax2.set_xlabel("Epoch", labelpad=15)  # Added padding to lift label
ax1.grid(True)
ax2.grid(True)
ax1.legend()
ax2.legend()

status_box = fig.text(0.5, 0, '', ha='center', va='bottom', fontsize=8, color='black')

def update_plot(frame):
    if not os.path.exists(RESULTS_CSV):
        status_box.set_text("Waiting for results.csv...")
        return

    try:
        df = pd.read_csv(RESULTS_CSV).fillna(0)
    except Exception as e:
        status_box.set_text(f"Error reading CSV: {e}")
        return

    mAP_col = 'metrics/mAP50'
    if mAP_col not in df.columns:
        if 'metrics/mAP50(B)' in df.columns:
            mAP_col = 'metrics/mAP50(B)'
        else:
            status_box.set_text("Waiting for mAP column...")
            return

    required_cols = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        mAP_col
    ]
    if not all(col in df.columns for col in required_cols):
        status_box.set_text("Waiting for full metrics in CSV...")
        return

    epochs = list(range(len(df)))
    train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    mAP = df[mAP_col]

    train_line.set_data(epochs, train_loss)
    val_line.set_data(epochs, val_loss)
    map_line.set_data(epochs, mAP)

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    fig.canvas.draw_idle()

    # ==== Logic starts only if a new epoch appears ====
    current_epoch = len(df) - 1
    if current_epoch != last_checked_epoch[0]:
        last_checked_epoch[0] = current_epoch

        status = "Training Smoothly"
        color = 'black'

        if current_epoch >= 5:
            overfit = val_loss.iloc[-1] > val_loss.iloc[-2] > val_loss.iloc[-3]
            if overfit:
                overfitting_counter[0] += 1
            else:
                overfitting_counter[0] = 0

            if overfitting_counter[0] >= 2 and not stopped[0]:
                status = "Overfitting Persisted - Training Stopped"
                color = 'darkred'
                pid = find_yolo_process()
                if pid:
                    os.kill(pid, signal.SIGTERM)
                    print(f"[MONITOR] Training process (PID {pid}) terminated.")
                    insights_log.append(f"Training stopped at epoch {current_epoch} due to persistent overfitting.\n")
                else:
                    print("[MONITOR] Could not locate YOLO process.")
                stopped[0] = True
                if not sent_overfit_alert[0]:
                    send_telegram_alert(f"⚠️ Overfitting detected for 5+ epochs. Training stopped for {worker_name}.")
                    sent_overfit_alert[0] = True

            elif mAP.iloc[-1] < 0.1 and not sent_lowmap_alert[0]:
                status = "Very Low mAP - Check Data"
                color = 'orange'
                send_telegram_alert(f"⚠️ Low mAP (<0.1) for {worker_name}. Check dataset/training.")
                sent_lowmap_alert[0] = True

            elif abs(train_loss.iloc[-1] - train_loss.iloc[-2]) < 0.001 and not sent_stagnant_alert[0]:
                status = "Stagnant Training"
                color = 'blue'
                send_telegram_alert(f"ℹ️ Training stagnant for {worker_name}. Consider adjusting parameters.")
                sent_stagnant_alert[0] = True

        log = f"Epoch {current_epoch} | Train Loss: {train_loss.iloc[-1]:.4f}, Val Loss: {val_loss.iloc[-1]:.4f}, mAP@0.5: {mAP.iloc[-1]:.4f} | STATUS: {status}"
        insights_log.append(log + '\n')

        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(f"YOLO Training Report - {os.path.basename(EXP_NAME)}\nGenerated: {datetime.now()}\n\n")
            f.writelines(insights_log)

        fig.savefig(os.path.join(SAVE_DIR, 'live_monitor.png'))
        status_box.set_text(log)
        status_box.set_color(color)

ani = FuncAnimation(fig, update_plot, interval=3000)
plt.tight_layout()
plt.show()
