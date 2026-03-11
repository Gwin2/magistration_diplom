param(
  [string]$ModelName = "uav_detector",
  [string]$MarFile = "uav_detector.mar",
  [string]$ManagementUrl = "http://localhost:8081",
  [string]$InitialWorkers = "1",
  [string]$Synchronous = "true"
)

$form = @{
  url             = $MarFile
  model_name      = $ModelName
  initial_workers = $InitialWorkers
  synchronous     = $Synchronous
}

Invoke-RestMethod -Method Post -Uri "$ManagementUrl/models" -Form $form | Out-Null
Write-Host "Model '$ModelName' registration request sent."
