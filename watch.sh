#!/bin/bash
inotifywait -r -e close_write,moved_to,create -m . |
while read -r directory events filename; do
    rsync -a . arccha@kogni3.ii.uj.edu.pl:~/fungus
done
