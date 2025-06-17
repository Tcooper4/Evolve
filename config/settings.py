# Upgrader Agent Settings
UPGRADER_SETTINGS = {
    'upgrade_interval': 24,  # hours between automatic upgrade checks
    'max_retries': 3,  # maximum number of retry attempts for failed upgrades
    'retry_delay': 300,  # seconds to wait between retry attempts
    'log_retention_days': 30,  # number of days to keep log files
    'memory_retention_days': 7,  # number of days to keep task memory
    'model_drift_threshold': 0.1,  # threshold for detecting model drift
    'version_check_interval': 3600,  # seconds between version checks
    'safe_mode': True,  # enable safe fallback mode
    'backup_before_upgrade': True,  # create backups before upgrades
    'notify_on_failure': True,  # send notifications on upgrade failures
    'max_concurrent_upgrades': 3,  # maximum number of concurrent upgrades
    'upgrade_timeout': 3600,  # maximum time (seconds) for an upgrade to complete
} 