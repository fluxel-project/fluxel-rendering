#Requires -Version 7.0
<#
.SYNOPSIS
Runs the release-only Windows GPU conformance gate and preserves its evidence.

.DESCRIPTION
This gate deliberately accepts only a clean x86_64-pc-windows-msvc checkout.
It injects HEAD into every fixture as FLUXEL_TEST_COMMIT and records the full
workspace ignored-test invocation under target/conformance/<sha>.  A failing
test command still leaves its manifest and combined log behind for review.

A green cargo exit alone does not pass this gate: it also requires that at
least $minimumCases ignored cases actually ran and that a GPU adapter was
recorded, because both can be lost while cargo still exits 0.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'x86_64-pc-windows-msvc'

# The number of `--ignored` cases this gate is expected to run.  It exists
# because a green cargo exit is not evidence that the hardware fixtures ran:
# libtest treats "`--ignored` matched nothing" as success, printing
# `running 0 tests` and exiting 0, so dropping or renaming one `#[ignore]` is a
# one-line edit that would otherwise be recorded here as a passing conformance
# run.  Raise this as the suite grows; lowering it is meant to be a deliberate
# act, which is the whole point.
$minimumCases = 89

function Get-CommandOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$File,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    $output = & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
    return @($output)
}

if (-not $IsWindows) {
    throw 'GPU conformance is a Windows/MSVC release gate; run this script on Windows.'
}

if ($env:CARGO_BUILD_TARGET -and $env:CARGO_BUILD_TARGET -ne $target) {
    throw "CARGO_BUILD_TARGET must be empty or $target; found '$env:CARGO_BUILD_TARGET'."
}

$rustcVerbose = Get-CommandOutput rustc '-vV'
$rustcHost = ($rustcVerbose | Where-Object { $_ -match '^host:' } | Select-Object -First 1)
if ($rustcHost -notmatch "^host:\s+$([regex]::Escape($target))$") {
    throw "rustc host must be $target; found '$rustcHost'."
}

$cargoVersion = (Get-CommandOutput cargo '--version') -join [Environment]::NewLine
# Lifted out of rustc -vV rather than left for the reader to parse, because the
# toolchain is part of what the artifact claims: this gate pins the host target
# but not the toolchain, so a run on nightly and a run on stable are
# indistinguishable unless the manifest says which one happened.
$toolchain = (
    ($rustcVerbose | Where-Object { $_ -match '^release:' } | Select-Object -First 1) `
        -replace '^release:\s*', ''
)
if (-not $toolchain) {
    throw 'rustc -vV reported no release line, so the manifest could not say which toolchain produced the run.'
}
$gitStatus = @(Get-CommandOutput git 'status' '--porcelain' '--untracked-files=all')
if ($gitStatus.Count -ne 0) {
    throw "conformance requires a clean worktree, including untracked files:`n$($gitStatus -join [Environment]::NewLine)"
}

$commit = ((Get-CommandOutput git 'rev-parse' 'HEAD') -join '').Trim()
if ($commit -notmatch '^[0-9a-f]{40}$') {
    throw "HEAD must resolve to a 40-character lowercase SHA; found '$commit'."
}

# Nothing above writes an artifact: a rejected checkout must not make itself dirty.
$artifactDirectory = Join-Path $PSScriptRoot "..\target\conformance\$commit"
$artifactDirectory = [IO.Path]::GetFullPath($artifactDirectory)
New-Item -ItemType Directory -Force -Path $artifactDirectory | Out-Null
$logPath = Join-Path $artifactDirectory 'cargo.log'
$manifestPath = Join-Path $artifactDirectory 'manifest.json'
$start = [DateTime]::UtcNow.ToString('o')
$env:FLUXEL_TEST_COMMIT = $commit
$env:RUST_TEST_THREADS = '1'

$gpu = @()
try {
    $gpu = @(Get-CimInstance -ClassName Win32_VideoController | ForEach-Object {
        [ordered]@{ name = $_.Name; driver_version = $_.DriverVersion; pnp_device_id = $_.PNPDeviceID }
    })
}
catch {
    $gpu = @([ordered]@{ collection_error = $_.Exception.Message })
}

# Test-support submit/completion faults are process-global by design.  The
# RUST_TEST_THREADS environment above serializes libtest without passing an
# unsupported --test-threads argument to Criterion benchmark targets.
$cargoArgs = @(
    'test', '--workspace', '--all-targets', '--all-features', '--locked',
    '--target', $target, '--', '--ignored', '--nocapture'
)
$exitCode = 1
try {
    & cargo @cargoArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
catch {
    $_ | Out-String | Tee-Object -FilePath $logPath -Append
    $exitCode = 1
}

$caseSummary = @()
$binarySummary = @()
$pendingCase = $null
if (Test-Path -LiteralPath $logPath) {
    foreach ($line in Get-Content -LiteralPath $logPath) {
        if ($line -match '^\s*Running (.+)$') {
            $binarySummary += $Matches[1]
        }
        elseif ($line -match '^test (.+) \.\.\. (ok|FAILED|ignored)$') {
            $caseSummary += [ordered]@{ case = $Matches[1]; result = $Matches[2] }
            $pendingCase = $null
        }
        elseif ($line -match '^test (\S+) \.\.\.') {
            # With --nocapture, fixture evidence may follow the test name and
            # libtest emits the result on a later line once that output ends.
            $pendingCase = $Matches[1]
        }
        elseif ($null -ne $pendingCase -and $line -match '^(ok|FAILED|ignored)$') {
            $caseSummary += [ordered]@{ case = $pendingCase; result = $Matches[1] }
            $pendingCase = $null
        }
    }
}

# A green cargo exit is not the same claim as "the hardware fixtures ran".  Two
# independent ways this run can exit 0 and still not be evidence, checked here
# rather than trusted to the exit code:
#   - libtest reports an --ignored filter that matched nothing as `running 0
#     tests` and exits 0, so dropping one #[ignore] silently removes a fixture;
#   - a run that recorded no adapter cannot be attributed to any GPU.
$shortfalls = @()
if ($caseSummary.Count -lt $minimumCases) {
    $shortfalls += (
        "only $($caseSummary.Count) ignored case(s) ran; this gate expects at " +
        "least $minimumCases"
    )
}
$adapters = @($gpu | Where-Object { $_.Contains('name') })
if ($adapters.Count -eq 0) {
    $shortfalls += 'no GPU adapter was recorded, so this run cannot be attributed to hardware'
}

$manifest = [ordered]@{
    schema = 1
    commit = $commit
    started_at_utc = $start
    finished_at_utc = [DateTime]::UtcNow.ToString('o')
    platform = [ordered]@{ os = 'Windows'; rustc_host = $target; cargo_target = $target }
    tools = [ordered]@{
        toolchain = $toolchain
        rustc_verbose = $rustcVerbose
        cargo_version = $cargoVersion
    }
    environment = [ordered]@{
        FLUXEL_TEST_COMMIT = $env:FLUXEL_TEST_COMMIT
        RUST_TEST_THREADS = $env:RUST_TEST_THREADS
    }
    hardware = $gpu
    command = [ordered]@{
        executable = 'cargo'
        arguments = $cargoArgs
        display = "cargo $($cargoArgs -join ' ')"
    }
    result = [ordered]@{
        exit_code = $exitCode
        minimum_cases = $minimumCases
        cases_observed = $caseSummary.Count
        adapters_observed = $adapters.Count
        shortfalls = $shortfalls
        log = 'cargo.log'
        test_binaries = $binarySummary
        test_cases = $caseSummary
    }
}
# The artifact directory is keyed by SHA, so a second run of the same revision
# lands on the first one's manifest.  That is the right key -- the gate refuses a
# dirty tree, so the same SHA is the same code -- but the *verdict* is not the
# same fact twice: a passing run and a later failing one at one revision are two
# readings, and overwriting the first silently is how a release ends up citing a
# verdict that the file no longer holds.  One generation is kept because one is
# what makes the loss visible; nothing here prunes beyond it.
if (Test-Path -LiteralPath $manifestPath) {
    Copy-Item -LiteralPath $manifestPath -Destination "$manifestPath.previous" -Force
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

if ($exitCode -ne 0) {
    [Console]::Error.WriteLine(
        "GPU conformance failed (exit $exitCode). Evidence: $artifactDirectory"
    )
    exit $exitCode
}

if ($shortfalls.Count -ne 0) {
    [Console]::Error.WriteLine(
        "GPU conformance ran green but is not evidence:`n  - $($shortfalls -join "`n  - ")`nEvidence: $artifactDirectory"
    )
    exit 1
}

Write-Host "GPU conformance passed ($($caseSummary.Count) cases, $($adapters.Count) adapter(s)). Evidence: $artifactDirectory"
