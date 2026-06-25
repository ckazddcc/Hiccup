#!/bin/sh

while [ ! -f ./USPEX_IS_DONE ]; do
   date >> log 
   matlab < USPEX.m  >> log
   sleep 300
done
