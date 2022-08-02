#!/bin/bash

# Parametrize all meshes that are in the directory "meshes" relative to the
# execution directory

mkdir parametrized_meshes

all_meshes=$(ls meshes/*.obj)
for i in $all_meshes
do
    echo "Parametrizing ${i}"
    ./uv_map "meshes/${i}" "parametrized_meshes/${i}"
done
