#!/bin/bash
# please read https://docs.google.com/document/d/1nGDRuOF5AWHaqgMHJvEYzDD7FHCFcLIfOD9spZe-TMg/edit?usp=sharing
git config --global pull.rebase false # probably only need this once
git add . # Make sure you have changes added and committed
git commit -m update
git checkout -b anderson-spring-2024-main main
git pull `git config --get remote.origin.url | awk -F \/ '{print $1}'`/data-301-student.git main
./fix_merge.sh
git checkout main
git merge --no-ff anderson-spring-2024-main -m fixed
git push origin main
