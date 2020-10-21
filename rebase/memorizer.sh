#!/bin/sh

rm -f memory.log
rm -f state.log
touch state.log

STOP="STOP"
SIGNAL=$(cat state.log)

bl_mem=$(free | grep Mem | awk {'print $3'})
bl_swap=$(free | grep Espazo | awk {'print $5'})
baseline=$(expr $bl_mem + $bl_swap)


while [ "$SIGNAL" != "$STOP" ]
do
i_mem=$(free | grep Mem | awk {'print $3'})
i_swap=$(free | grep Espazo | awk {'print $5'})
iter=$(expr $i_mem + $i_swap)

expr $iter - $baseline >> memory.log

sleep 1

SIGNAL=$(cat state.log)

done
