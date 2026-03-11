param(
  [string]$Config = "configs/experiments/yolos_tiny.yaml",
  [string]$RunsDir = "runs",
  [string]$ReportsDir = "reports",
  [string]$MetadataCsv = ""
)

$ErrorActionPreference = "Stop"

uav-vit train --config $Config
uav-vit evaluate --config $Config --split test
uav-vit summarize --runs-dir $RunsDir --output-dir $ReportsDir

if ($MetadataCsv -ne "") {
  uav-vit analyze-conditions --config $Config --metadata-csv $MetadataCsv --column weather --split test
  uav-vit analyze-conditions --config $Config --metadata-csv $MetadataCsv --column quality --split test
  uav-vit analyze-conditions --config $Config --metadata-csv $MetadataCsv --column maneuver --split test
}

Write-Host "Pipeline completed."
