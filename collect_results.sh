rm -f results.zip
zip -r results.zip  . -i "*.py" "*.pdf" "*.md" "*.txt" ".gitignore" -x "*.ipynb_checkpoints*"
