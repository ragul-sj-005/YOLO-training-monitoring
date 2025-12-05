# YOLOv8 Live Training Monitor

A **real-time training monitor for YOLOv8** that tracks training progress, detects overfitting, identifies stagnant training, and sends Telegram alerts. Visualize **train/validation loss** and **mAP@0.5** during training and take action automatically if training behaves abnormally.

---

## 🚀 Features

- Live plotting of training and validation losses.
- Real-time monitoring of mAP@0.5 to assess detection accuracy.
- Automatic detection of overfitting:
  - Stops training if validation loss increases for multiple consecutive epochs.
- Detection of low mAP:
  - Alerts if the model fails to detect objects effectively.
- Stagnant training detection:
  - Alerts if training loss stops decreasing.
- Telegram notifications for all critical events.
- Automatic termination of YOLO training process (optional) if severe overfitting persists.
- Generates training reports and snapshot images of live metrics.
