param(
  [string]$ModelName = "uav_detector",
  [Parameter(Mandatory = $true)][string]$ImagePath,
  [string]$InferenceUrl = "http://localhost:8080"
)

Invoke-RestMethod -Method Post -Uri "$InferenceUrl/predictions/$ModelName" -InFile $ImagePath -ContentType "application/octet-stream"
