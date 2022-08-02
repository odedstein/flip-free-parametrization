files = dir(fullfile('.','*.obj'));
for k = 1:length(files)
    file_name = files(k).name;
    fprintf('Processing %s.\n', file_name);
    [V,F] = readOBJ(file_name);
    
    delete_mesh = false;
    if(sum(F(:)<=0) ~= 0)
        delete_mesh = true;
    elseif(sum(is_vertex_nonmanifold(F))>0)
        delete_mesh = true;
    elseif(size(nonmanifold_edges(F),1)>0)
        delete_mesh = true;
    elseif(0.5*min(doublearea(V,F)) < eps)
        delete_mesh = true;
    elseif(size(V,1)+size(F,1)-size(edges(F),1)~=1)
        delete_mesh = true;
    end
    
    if(delete_mesh)
        fprintf('- Deleted %s.\n', file_name);
        delete(file_name);
    end
end