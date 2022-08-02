function [V,F] = mesh_from_png(path,tol)
%MESH_FROM_PNG Take the png at path and save as mesh. The path at path must
%end in .png
%This function requires gptoolbox to run
%(https://github.com/alecjacobson/gptoolbox)

%the tol values used to create the meshes in this folder are:
%octopus: 3
%mouth: 3
%robot: 5

pathEls = split(path,'.');
if(~strcmp(pathEls(end),'png') && ~strcmp(pathEls(end),'PNG'))
    error('path must end in png');
end
name = join(pathEls{1:end-1},'.');

[V,F] = bwmesh(path, 'Tol',tol, 'SmoothingIters',3);
V = V - mean(V);
bb = bounding_box(V);
scale = norm(max(bb) - min(bb));
V = V / scale;

objPath = [name '.obj'];
writeOBJ(objPath, V, F);

end

