#!/usr/bin/env bash
set -uo pipefail

if (( $# == 0 )); then
    printf 'Usage: %s [--delay SECS] URL [URL ...]\n' "$0" >&2
    exit 2
fi

# Seconds to wait before a repeat request to a domain already contacted in this run. Batches
# of new links are usually all from one vendor, and hammering that vendor gets this runner
# throttled or blocked - which costs far more than the wait. The first hit per domain is
# immediate, so checking a handful of unrelated URLs stays fast.
delay=3
if [[ "${1:-}" == --delay ]]; then
    delay=$2
    shift 2
elif [[ "${1:-}" == --delay=* ]]; then
    delay=${1#--delay=}
    shift
fi

status=0
seen=" "

for url in "$@"; do
    case "$url" in
        http://*|https://*) ;;
        *)
            printf 'INVALID %s\n' "$url" >&2
            status=1
            continue
            ;;
    esac

    # host = everything between the scheme and the first following slash
    host=${url#*://}
    host=${host%%/*}
    if [[ "$seen" == *" $host "* ]]; then
        sleep "$delay"
    else
        seen+="$host "
    fi

    curl_args=(
        --location
        --fail
        --silent
        --show-error
        --retry 2
        --retry-all-errors
        --retry-delay "$delay"
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
