cd /data/local/tmp/lt
export LD_LIBRARY_PATH=/data/local/tmp/lt:/vendor/lib64:/system/lib64 CLASSPATH=/data/local/tmp/lt/classes_mem.dex
# usage: MODELS=.. RUNS="backend:spec:tag ..." PROMPT=p4k.txt MAXTOK=4096 sh mem4.sh
for m in $MODELS; do for r in $RUNS; do b=${r%%:*}; rest=${r#*:}; sp=${rest%%:*}; tag=${rest#*:}
SPEC=$sp app_process -Djava.library.path=/data/local/tmp/lt /data/local/tmp/lt Bench /data/local/tmp/lt/$m.litertlm $b ${MAXTOK:-4096} ${PROMPT:-p4k.txt} 1 1 150 > m4_${m}_${b}_${sp}_$tag.log 2>&1 &
P=$!; f=m4_${m}_${b}_${sp}_$tag.mem; rm -f $f
while kill -0 $P 2>/dev/null; do dumpsys meminfo $P 2>/dev/null | grep -E '^ +(GL mtrack|TOTAL PSS|TOTAL RSS)' | tr -s ' ' | tr '\n' '|' >> $f; echo >> $f; sleep 3; done
echo "== $m backend $b spec $sp $tag"; grep -E "^init|^prefill|^decode|^peak|^MEM" m4_${m}_${b}_${sp}_$tag.log | cut -c1-130
echo "peak GL $(grep -o 'GL mtrack [0-9]*' $f | awk '{print $3}' | sort -n | tail -1) KB, peak PSS $(grep -o 'TOTAL PSS: [0-9]*' $f | awk '{print $3}' | sort -n | tail -1) KB, peak RSS $(grep -o 'TOTAL RSS: [0-9]*' $f | awk '{print $3}' | sort -n | tail -1) KB"
cp decode_out.txt o4_${m}_${b}_${sp}_$tag.txt
done; done
