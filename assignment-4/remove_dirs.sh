#!/bin/bash
# Removes the weirdly named directories so that Windows can use them.

CHUNK_DIR="./chunks"  # not used
NUM_CORES=1   # Change this to the number of CPU cores you want to utilize

process_chunk() { 
  # Read each line from the file and rename the files
  while IFS= read -r file
  do
    if [ -d "$file" ]; then
      mv -fv ${file} "${file}-temp"
      mv -fv ${file}-temp/* ${file}-temp/..
      rm -r ${file}-temp
    else
      echo "Dir not found: $file"
    fi
  done < $1
}

# Find all invalid files, save paths to file
find . -type d -name *. > temp2.txt

# Separate file into chunks
mkdir chunks
rm ./chunks/*
split -n $NUM_CORES temp2.txt ./chunks/chunk_

# Process chunks
for chunk_file in ./chunks/chunk_*; do
  echo "Processing $chunk_file"
  process_chunk $chunk_file &
done

# Remove temp files/dirs
rm -r ./chunks
rm temp2.txt

# Wait for all background jobs to finish
wait
echo ""
echo "All chunks processed"
