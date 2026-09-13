# Installer troubleshooting

- SmartScreen: verify the release signature and use the documented fixture installer only.
- Setup timeout: retain `clean-install-evidence.json`, launcher logs, worker logs, ports, and process-cleanup evidence.
- Hash failure: retry only the corrupted fixture component; never substitute a developer checkout.
- Uninstall: retain data for app-only uninstall. Full owned-data uninstall must retain unmarked user files and unrelated Ollama models.
