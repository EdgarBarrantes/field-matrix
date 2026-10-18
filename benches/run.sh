#!/bin/sh
# Runs `cargo bench` under the same conditions every time, for results worth comparing:
#   1. waits until the machine is idle: at least 90% of CPU time idle over 30 seconds;
#   2. pins the benchmark to one performance core (both its hardware threads), so it never lands
#      on a slower efficiency core and nothing else shares that core;
#   3. reports how idle the other cores stayed during the run (below 90%: don't trust the run).
# Arguments go to `cargo bench`, e.g. `benches/run.sh -- --save-baseline before`.
# Changes no system settings; on machines without separate core types it runs unpinned.
set -eu
cd "$(dirname "$0")/.."

# Total and idle CPU time (idle + iowait) from /proc/stat, for all CPUs or for one ("cpu3").
cpu_times() {
    awk -v cpu="${1:-cpu}" '$1 == cpu { idle = $5 + $6; total = 0; for (i = 2; i <= NF; i++) total += $i; print total, idle }' /proc/stat
}

# Percentage of CPU time that was idle between two `cpu_times` samples.
idle_percent() {
    echo "$1 $2" | awk '{ printf "%d", 100 * ($4 - $2) / ($3 - $1) }'
}

until
    before=$(cpu_times)
    sleep 30
    idle=$(idle_percent "$before" "$(cpu_times)")
    [ "$idle" -ge 90 ]
do
    echo "waiting: CPU ${idle}% idle over the last 30 s (need 90%)" >&2
done
echo "idle: ${idle}% over 30 s, $(date +%T)" >&2

pin=""
if [ -r /sys/devices/cpu_core/cpus ]; then
    # The performance core that holds the third performance CPU (not the first core, which
    # handles more interrupts), with both its hardware threads.
    cpu=$(($(cut -d- -f1 /sys/devices/cpu_core/cpus) + 2))
    pin=$(cat "/sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list")
    echo "pinned to performance core CPUs $pin" >&2
fi

others_before=""
for c in $(seq 0 $(($(nproc --all) - 1))); do
    case ",$pin," in *",$c,"* | *",$c-"* | *"-$c,"*) continue ;; esac
    others_before="$others_before cpu$c:$(cpu_times "cpu$c" | tr ' ' ,)"
done

if [ -n "$pin" ]; then
    taskset -c "$pin" cargo bench "$@"
else
    cargo bench "$@"
fi

total_before=0 idle_before=0 total_after=0 idle_after=0
for entry in $others_before; do
    c=${entry%%:*}
    times=${entry#*:}
    total_before=$((total_before + ${times%,*}))
    idle_before=$((idle_before + ${times#*,}))
    set -- $(cpu_times "$c")
    total_after=$((total_after + $1))
    idle_after=$((idle_after + $2))
done
others=$(idle_percent "$total_before $idle_before" "$total_after $idle_after")
if [ "$others" -ge 90 ]; then
    echo "other cores ${others}% idle during the run: ok" >&2
    verdict="ok"
else
    echo "WARNING: other cores only ${others}% idle during the run: something else ran, don't trust it" >&2
    verdict="DISTURBED"
fi

# For benches/summarize.py.
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
mkdir -p target/criterion
echo "pinned to CPUs ${pin:-none}, governor $governor, ${idle}% idle before, other cores ${others}% idle during ($verdict)" \
    > target/criterion/conditions.txt
