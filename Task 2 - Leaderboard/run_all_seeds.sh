#!/bin/bash
# runs all 9 (variant, seed) jobs, two at a time
cd "$(dirname "$0")"
jobs_list="none:0 past:0 pastfuture:0 none:1 past:1 pastfuture:1 none:2 past:2 pastfuture:2"
for j in $jobs_list; do
  v=${j%%:*}; s=${j##*:}
  while [ $(pgrep -fc "task2.py run") -ge 2 ]; do sleep 5; done
  nohup python3 task2.py run --variant $v --seed $s > logs/${v}_s${s}.log 2>&1 &
  sleep 2
done
wait
