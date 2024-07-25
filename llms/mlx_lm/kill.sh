#!/usr/bin/env bash
ps -ef|grep 'mlx_lm.server'|awk '{print $2}'|xargs kill -9