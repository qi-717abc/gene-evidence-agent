# Daily education mode. The backend reads the existing API configuration.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$geaExisting = $null
try { $geaExisting = Invoke-RestMethod 'http://127.0.0.1:8765/api/health' -TimeoutSec 3 } catch { }
if ($geaExisting) {
    if ($geaExisting.instance.application -ne 'gene-evidence-agent') {
        throw 'Port 8765 is serving an unidentified application. No process was changed.'
    }
    Write-Output "Agent already running: mode=$($geaExisting.mode), pid=$($geaExisting.instance.pid)"
    Write-Output "Education model: $($geaExisting.education.chat.status)"
    Write-Output 'http://127.0.0.1:8765/'
    Write-Output 'To change mode, stop that instance in its launch window before running this script again.'
    return
}
$env:GEA_WEB_MODE = 'live'
$env:GEA_WEB_MODEL_ENABLED = 'true'
$env:GEA_MODEL_ENABLED = 'true'
$env:GEA_WEB_EDUCATION_CHAT_ENABLED = 'true'
$env:GEA_WEB_CHAT_ENABLED = 'false'
$env:GEA_WEB_SYNTHESIS_ENABLED = 'false'
$env:GEA_WEB_MODEL_BUDGET_STAGE = 'p5'
$env:GEA_LOG_LEVEL = 'WARNING'
$env:GEA_WEB_PORT = '8765'
& "$PSScriptRoot\.venv\Scripts\python.exe" -u -m gene_evidence_agent.web_server
