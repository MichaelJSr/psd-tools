#!/bin/zsh
# ab.sh <tag> <proto src> [env assignments...]: base vs proto render per card, interleaved; outputs out/ab_<tag>.txt
export PYTHONDONTWRITEBYTECODE=1
PY=~/.venvs/ac-py314-fork/bin/python
B=/Users/michaelsrouji/Documents/Tools/wt/research-speed/_bench
BASE=/Users/michaelsrouji/Documents/Tools/wt/research-speed/src
tag=$1; proto=$2; shift 2
: > $B/out/ab_$tag.txt
i=0
while IFS= read -r c; do
  i=$((i+1))
  if [[ ! -f $B/out/arr/base_$i.npz ]]; then
    PYTHONPATH=$BASE $PY $B/render_save.py "$c" $B/out/arr/base_$i.npz > $B/out/arr/base_$i.json
  fi
  b=$(cat $B/out/arr/base_$i.json)
  p=$(env "$@" PYTHONPATH=$proto $PY $B/render_save.py "$c" $B/out/arr/${tag}_$i.npz 2>>$B/out/ab_$tag.err)
  d=$($PY $B/cmp.py $B/out/arr/base_$i.npz $B/out/arr/${tag}_$i.npz)
  echo "$i	$b	$p	$d" >> $B/out/ab_$tag.txt
done < ${CARDS:-$B/cards20.txt}
echo done $tag
