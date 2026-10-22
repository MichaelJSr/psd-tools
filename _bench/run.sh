#!/bin/zsh
# run.sh <mode> <tag> [prof]: every card of cards20.txt in its own process, sequentially
export PYTHONPATH=${SRC:-/Users/michaelsrouji/Documents/Tools/wt/research-speed/src} PYTHONDONTWRITEBYTECODE=1
PY=~/.venvs/ac-py314-fork/bin/python
B=/Users/michaelsrouji/Documents/Tools/wt/research-speed/_bench
mkdir -p $B/out/$2
: > $B/out/$2.jsonl
i=0
while IFS= read -r c; do
  i=$((i+1))
  if [[ "$3" == prof ]]; then
    $PY $B/bench_one.py $1 "$c" $B/out/$2/$i.prof >> $B/out/$2.jsonl 2>> $B/out/$2.err
  else
    $PY $B/bench_one.py $1 "$c" >> $B/out/$2.jsonl 2>> $B/out/$2.err
  fi
done < ${CARDS:-$B/cards20.txt}
echo done $2
