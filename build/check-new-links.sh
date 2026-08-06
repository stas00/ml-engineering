#!/usr/bin/env bash
set -uo pipefail

if (( $# == 0 )); then
    printf 'Usage: %s URL [URL ...]\n' "$0" >&2
    exit 2
fi

status=0

for url in "$@"; do
    case "$url" in
        http://*|https://*) ;;
        *)
            printf 'INVALID %s\n' "$url" >&2
            status=1
            continue
            ;;
    esac

    curl_args=(
        --location
        --fail
        --silent
        --show-error
        --retry 2
        --retry-all-errors
        --max-time 30
        --output /dev/null
        --write-out '%{http_code} %{url_effective}'
    )

    if result=$(curl "${curl_args[@]}" "$url" 2>&1); then
        printf 'LIVE %s -> %s\n' "$url" "$result"
    elif result=$(curl --http1.1 "${curl_args[@]}" "$url" 2>&1); then
        printf 'LIVE %s -> %s\n' "$url" "$result"
    else
        printf 'DEAD %s -> %s\n' "$url" "$result" >&2
        status=1
    fi
done

exit "$status"
