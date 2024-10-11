#!/bin/bash

function rand(){
    min=$1
    max=$(($2 - $min + 1))
    num=$(($RANDOM + 1000000000))
    echo $(($num % $max + $min))
}

function alloc_port() {
    port=0
    while [ ${port} == 0 ] 
    do
        tmp=$(rand $1 $2)
        if ss -tuln | grep ":${tmp} " > /dev/null
        then
            continue
        fi
        port=${tmp}
    done
    echo $(($port))
}