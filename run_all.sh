#!/bin/bash

apps="deformation uv_maps volumetric_map unflipping_triangles limitations comparison_with_previous"

cd build/applications

for i in $apps
do
    cd $i
    echo "Running $i..."
    ./$i
    cd ..
done

cd ../..

