# Checkpoint hook configuration file
checkpoint_config = dict(interval=1)  # The saving interval is 1
# Log configuration information
log_config = dict(
    interval=100,  # The interval for printing logs
    hooks=[
        dict(type='TextLoggerHook'),  # A text logger for recording the training process
    ])

launcher = 'pytorch'
log_level = 'INFO'  # Log output level
resume_from = None  # Restore checkpoints from the given path. Training mode will resume training from the epoch saved in the checkpoint
load_from = None
workflow = [('train', 2)]  # Runner workflow
